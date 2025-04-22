import os
import json
import torch
import argparse
import time
import gc
from PIL import Image
from torch import nn
from tqdm import tqdm

# diffusers 相关
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    HunyuanDiTPipeline
)
from diffusers.models.normalization import AdaGroupNorm

# 模型相关
from model import NoiseTransformer, SVDNoiseUnet

class NPNet(nn.Module):
    def __init__(self, model_id="SDXL", device="cuda", pretrained_path=""):
        """
        如果 pretrained_path 是具体的 .pth 文件路径，则会加载其中的模型权重和 alpha, beta，
        否则使用空模型（随机初始化）。
        """
        super().__init__()
        assert model_id in ["SDXL", "DreamShaper", "DiT"], "Unsupported model_id."

        self.device = device
        self.model_id = model_id

        # 初始化组件
        self.unet_svd = SVDNoiseUnet(resolution=128).to(device).to(torch.float32)
        self.unet_embedding = NoiseTransformer(resolution=128).to(device).to(torch.float32)
        if self.model_id == 'DiT':
            self.text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(device).to(torch.float32)
        else:
            self.text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(device).to(torch.float32)

        # 初始化可学习的 alpha 和 beta
        self._alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self._beta = nn.Parameter(torch.tensor(0.0, device=self.device))

        # 如果检测到 .pth，则从该文件加载预训练权重
        if pretrained_path and pretrained_path.endswith(".pth"):
            state_dict = torch.load(pretrained_path, map_location=device)
            self.unet_svd.load_state_dict(state_dict["unet_svd"])
            self.unet_embedding.load_state_dict(state_dict["unet_embedding"])
            self.text_embedding.load_state_dict(state_dict["embeeding"])
            # 强制 reshape 加载的参数以匹配当前参数形状，然后使用 .data.copy_()
            self._alpha.data.copy_(state_dict["alpha"].to(device).reshape(self._alpha.data.shape))
            self._beta.data.copy_(state_dict["beta"].to(device).reshape(self._beta.data.shape))
            print(f"Loaded pretrained NPNet weights from {pretrained_path}")

    def forward(self, initial_noise, prompt_embeds):
        """
        initial_noise: [B, C, H, W]
        prompt_embeds: [B, seq_len, emb_dim] 或展平后 [B, ...]
        """
        # 1) 先将 prompt_embeds 展平为 [B, -1]
        prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)

        # 2) 计算 text_emb
        text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

        # 3) 通过 NoiseTransformer 进行处理
        encoder_hidden_states_embedding = initial_noise + text_emb
        golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

        # 4) 通过 SVDNoiseUnet 得到最终 golden noise
        golden_noise = (
            self.unet_svd(initial_noise.float())
            + (2 * torch.sigmoid(self._alpha) - 1) * text_emb
            + self._beta * golden_embedding
        )
        return golden_noise


def generate_prompt_noise_pairs(
    text_dataset_path,
    output_file="prompt_noise_pairs.pt",
    batch_size=4,
    num_samples=8,
    model_id="SDXL",
    pretrained_path="sdxl.pth",
    device="cuda",
    save_interval=100,  # 每处理多少批次保存一次
    start_idx=0,        # 可选：从指定索引处开始处理
    reset_interval=500  # 每处理多少批次重置一次管道
):
    """
    读取文本提示数据集，并生成 golden noise 数据对。
    改进版本支持分批保存和内存管理。
    """
    # 1) 读取文本提示
    prompts = []
    if text_dataset_path.endswith(".json"):
        if not os.path.exists(text_dataset_path):
            print(f"Text JSON file not found: {text_dataset_path}")
            return
        with open(text_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 假设 JSON 文件中有 "annotations" 键，每个元素都是字典且包含 "caption" 字段
        for ann in data.get("annotations", []):
            caption = ann.get("caption", "").strip()
            if caption:
                prompts.append(caption)
    else:
        if not os.path.exists(text_dataset_path):
            print(f"Text dataset not found: {text_dataset_path}")
            return
        with open(text_dataset_path, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    if not prompts:
        print("No prompts found.")
        return

    # 如果有预先保存的文件，加载它并获取已处理的提示数量
    if os.path.exists(output_file) and start_idx == 0:
        try:
            saved_data = torch.load(output_file)
            num_saved_prompts = len(saved_data.get("prompts", []))
            if num_saved_prompts > 0:
                start_idx = num_saved_prompts // num_samples
                print(f"Found existing data with {num_saved_prompts} prompts. Continuing from batch {start_idx}.")
        except Exception as e:
            print(f"Error loading existing file: {e}")
            # 如果加载失败，从头开始
            start_idx = 0

    # 确保 start_idx 不超过提示总数
    if start_idx >= len(prompts):
        print(f"Start index {start_idx} exceeds total prompts {len(prompts)}. Nothing to process.")
        return

    print(f"Starting from prompt index {start_idx}/{len(prompts)}")
    
    # 为增量保存创建目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 初始化跟踪变量
    all_prompts = []
    all_random_noises = []
    all_golden_noises = []
    
    # 用于进度监控的变量
    last_time = time.time()
    last_batch = start_idx
    pipe = None
    net = None
    
    def init_or_reset_models():
        """初始化或重置模型，释放GPU内存"""
        nonlocal pipe, net
        
        # 清理之前的模型（如果存在）
        if pipe is not None:
            del pipe
        if net is not None:
            del net
            
        # 强制垃圾收集和清空CUDA缓存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 初始化文本编码管线
        if model_id == "DiT":
            pipe = HunyuanDiTPipeline.from_pretrained(
                "Tencent-Hunyuan/hunyuan-DiT",
                torch_dtype=torch.float16
            ).to(device)
        else:  # SDXL or DreamShaper
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16
            ).to(device)
            
        pipe.enable_model_cpu_offload()
        
        # 初始化NPNet模型
        net = NPNet(model_id=model_id, device=device, pretrained_path=pretrained_path).to(device)
        net.eval()
        
        print(f"Models initialized/reset. Current GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # 初始化模型
    init_or_reset_models()
    
    total_prompts = len(prompts)
    # 我们将从start_idx开始处理
    try:
        for batch_start in tqdm(range(start_idx, total_prompts, batch_size), 
                               desc="Processing prompts", unit="batch",
                               initial=start_idx, total=total_prompts // batch_size):
            # 监控处理速度，如果变慢就清理缓存
            current_time = time.time()
            batch_time = current_time - last_time
            batches_processed = batch_start - last_batch
            
            if batches_processed > 0:
                batches_per_second = batches_processed / batch_time
                # 如果速度明显下降（低于0.5批/秒或者每批超过2秒），清理缓存
                if batches_per_second < 0.5:
                    print(f"\nProcessing speed decreased to {batches_per_second:.2f} batches/s. Cleaning cache...")
                    torch.cuda.empty_cache()
                    
                # 如果速度极其缓慢（低于0.1批/秒或每批超过10秒），重置模型
                if batches_per_second < 0.1 or (batch_start > 0 and batch_start % reset_interval == 0):
                    print(f"\nProcessing speed critically low or reset interval reached. Reinitializing models...")
                    init_or_reset_models()
            
            # 更新监控变量
            last_time = current_time
            last_batch = batch_start
            
            end_idx = min(batch_start + batch_size, total_prompts)
            batch_prompts = prompts[batch_start:end_idx]
            
            # 对于每个prompt，复制num_samples次
            repeated_prompts = []
            for p in batch_prompts:
                repeated_prompts.extend([p] * num_samples)
            
            N = len(repeated_prompts)
            
            # 生成随机噪声
            random_noise = torch.randn(N, 4, 128, 128, device=device, dtype=torch.float32)
            
            # 编码提示文本
            with torch.no_grad():
                if model_id == "DiT":
                    # HunyuanDiT可能有不同的编码方式
                    text_embeds = pipe.encode_prompt(
                        repeated_prompts, device=device)
                else:
                    # SDXL的编码方式
                    text_embeds, pooled_emb, _, _ = pipe.encode_prompt(
                        repeated_prompts, device=device, do_classifier_free_guidance=True
                    )
            
            # 生成golden noise
            with torch.no_grad():
                golden_noise = net(random_noise, text_embeds)
            
            # 收集结果
            all_prompts.extend(repeated_prompts)
            all_random_noises.append(random_noise.cpu())
            all_golden_noises.append(golden_noise.cpu())
            
            # 定期保存结果（每处理save_interval个批次）
            if (batch_start > 0 and (batch_start % save_interval == 0)) or batch_start + batch_size >= total_prompts:
                # 创建增量保存的文件名
                temp_file = f"{os.path.splitext(output_file)[0]}_temp_{batch_start}.pt"
                
                # 合并当前批次的数据
                random_noises_tensor = torch.cat(all_random_noises, dim=0)
                golden_noises_tensor = torch.cat(all_golden_noises, dim=0)
                
                # 保存当前批次数据
                torch.save({
                    "prompts": all_prompts,
                    "random_noises": random_noises_tensor,
                    "golden_noises": golden_noises_tensor
                }, temp_file)
                
                print(f"\nSaved intermediate results to {temp_file}")
                print(f"Processed {len(all_prompts)} prompts so far ({batch_start}/{total_prompts} batches)")
                print(f"Current memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                # 清理内存，只保留必要的数据
                all_prompts = []
                all_random_noises = []
                all_golden_noises = []
                torch.cuda.empty_cache()
                
                # 尝试合并所有临时文件（如果需要）
                if batch_start + batch_size >= total_prompts:
                    merge_temp_files(output_file, output_dir)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        if all_prompts:
            temp_file = f"{os.path.splitext(output_file)[0]}_interrupted_{batch_start}.pt"
            random_noises_tensor = torch.cat(all_random_noises, dim=0) if all_random_noises else torch.tensor([])
            golden_noises_tensor = torch.cat(all_golden_noises, dim=0) if all_golden_noises else torch.tensor([])
            
            torch.save({
                "prompts": all_prompts,
                "random_noises": random_noises_tensor,
                "golden_noises": golden_noises_tensor
            }, temp_file)
            print(f"Saved progress to {temp_file}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        # 保存当前进度
        if all_prompts:
            error_file = f"{os.path.splitext(output_file)[0]}_error_{batch_start}.pt"
            random_noises_tensor = torch.cat(all_random_noises, dim=0) if all_random_noises else torch.tensor([])
            golden_noises_tensor = torch.cat(all_golden_noises, dim=0) if all_golden_noises else torch.tensor([])
            
            torch.save({
                "prompts": all_prompts,
                "random_noises": random_noises_tensor,
                "golden_noises": golden_noises_tensor
            }, error_file)
            print(f"Saved progress before error to {error_file}")
        raise
    
    print(f"Processing completed successfully!")


def merge_temp_files(output_file, output_dir):
    """合并所有临时文件为一个最终文件"""
    print("Merging temporary files...")
    
    all_prompts = []
    all_random_noises = []
    all_golden_noises = []
    
    # 查找所有临时文件
    temp_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(os.path.splitext(output_file)[0]) + "_temp_") and f.endswith(".pt")]
    temp_files.sort(key=lambda x: int(x.split("_temp_")[1].split(".")[0]))
    
    # 依次加载并合并数据
    for temp_file in temp_files:
        file_path = os.path.join(output_dir, temp_file)
        try:
            data = torch.load(file_path)
            all_prompts.extend(data["prompts"])
            all_random_noises.append(data["random_noises"])
            all_golden_noises.append(data["golden_noises"])
            
            # 删除临时文件
            os.remove(file_path)
            print(f"Merged and removed {temp_file}")
        except Exception as e:
            print(f"Error merging {temp_file}: {e}")
    
    # 保存最终结果
    if all_prompts:
        random_noises_tensor = torch.cat(all_random_noises, dim=0)
        golden_noises_tensor = torch.cat(all_golden_noises, dim=0)
        
        torch.save({
            "prompts": all_prompts,
            "random_noises": random_noises_tensor,
            "golden_noises": golden_noises_tensor
        }, output_file)
        
        print(f"Merged all data into {output_file}")
        print(f"Total prompts: {len(all_prompts)}")
    else:
        print("No data to merge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompt-noise pairs via a pretrained NPNet.")
    parser.add_argument("--text_dataset_path", type=str, default="my_golden_noise/dataset/annotations/captions_val2014.json", help="Path to text prompt dataset (JSON or txt).")
    parser.add_argument("--output_file", type=str, default="prompt_noise_pairs.pt", help="Output .pt file path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size: number of unique prompts processed simultaneously")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of random noises per prompt")
    parser.add_argument("--model_id", type=str, default="SDXL", help="Model ID for NPNet")
    parser.add_argument("--pretrained_path", type=str, default="sdxl.pth", help="Path to the pretrained NPNet weights (.pth)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save intermediate results every N batches")
    parser.add_argument("--start_idx", type=int, default=0, help="Start processing from this prompt index")
    parser.add_argument("--reset_interval", type=int, default=5000, help="Reset models every N batches")

    args = parser.parse_args()

    generate_prompt_noise_pairs(
        text_dataset_path=args.text_dataset_path,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        model_id=args.model_id,
        pretrained_path=args.pretrained_path,
        device=args.device,
        save_interval=args.save_interval,
        start_idx=args.start_idx,
        reset_interval=args.reset_interval
    )

#python get_dataset.py --save_interval 50 --reset_interval 200 --start_idx 16200