# Golden Noise for Diffusion Models - 序列演化实现

本项目基于论文 "Golden Noise for Diffusion Models: A Learning Framework" (arXiv:2411.09502) 的概念，并探索了一种**序列化**生成“黄金噪声”的方法。项目包含以下功能：

1.  **提取 Prompts:** 从标准数据集中提取文本提示。
2.  **生成噪声序列数据集 (NPD):** 使用基础扩散模型（如 SDXL）和 DDIM 逆向过程，为每个 prompt 生成一个从初始噪声 $x_T$ 逐步演化到 $x'_n$ 的噪声序列 $[x'_1, ..., x'_n]$。
3.  **训练模型:**
	* 训练原始论文提出的 NPNet 模型（作为基线）。
	* 训练基于 Transformer 的序列模型 (`NoiseSequenceTransformer`) 来预测噪声序列。
	* 训练基于 RNN (GRU) 的改进序列模型 (`NoiseSequenceRNN_v2`) 来预测噪声序列（包含 ResNet 风格 CNN、FiLM 条件和残差预测选项）。
4.  **推理:** 使用训练好的模型（目前主要支持原始 NPNet）生成黄金噪声并结合基础扩散模型生成图像。
5.  **评估:** 使用评估脚本（目前主要支持原始 NPNet）比较标准噪声和黄金噪声生成的图像质量（例如使用 HPSv2）。

## 项目结构

```
CS5340_PROJECT/
├── data/                     # 数据集目录
│   ├── pickapic_prompts.txt  # 提取的 prompts 文件 (示例)
│   └── npd_sequence_dataset_sdxl/ # 生成的序列数据集 (示例)
│       ├── sequences/        # 保存的噪声序列文件 (.pt)
│       └── metadata.csv      # 元数据
├── model/                    # 模型定义
│   ├── __init__.py
│   ├── Attention.py
│   ├── NoiseTransformer.py
│   ├── SVDNoiseUnet.py
│   ├── npnet.py              # 原始 NPNet 模型 + CrossAttention
│   ├── seq_model.py          # Transformer 序列模型
│   └── rnn_seq_model_v2.py   # RNN 序列模型 (v1 可能在此或单独文件)
├── scripts/                  # 辅助脚本
│   └── extract_prompts.py    # 提取 prompts 脚本
├── src/                      # 主要源码
│   ├── generate_npd.py       # 生成 NPD 序列数据集脚本
│   ├── dataset.py            # PyTorch Dataset 和 DataLoader
│   ├── train.py              # 训练原始 NPNet 脚本
│   ├── train_seq_model.py    # 训练 Transformer 序列模型脚本
│   ├── train_rnn_model_v2.py # 训练 RNN v2 序列模型脚本
│   ├── inference.py          # 使用 NPNet 生成图像脚本
│   └── evaluate.py           # 评估 NPNet 脚本
├── requirements.txt          # Python 依赖
└── README.md                 # 本文档
```

## 环境设置

1.  **前提条件:**
	* Python 3.8+
	* PyTorch (推荐 CUDA 版本)
	* CUDA Toolkit (如果使用 GPU)

2.  **克隆仓库:**
	```bash
	git clone <your-repo-url>
	cd CS5340_PROJECT
	```

3.  **安装依赖:**
	强烈建议使用 Conda 或 venv 创建虚拟环境。
	```bash
	# conda create -n golden_env python=3.8 -c conda-forge -y
	# conda activate golden_env
	pip install -r requirements.txt
	```

## `requirements.txt` (示例)

```txt
torch>=1.13.0
torchvision
torchaudio
# 选择与你的 PyTorch 和 CUDA 匹配的版本
# pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # (例如 CUDA 11.8)

diffusers>=0.21.0 # 确保版本支持 SDXL 等模型
transformers>=4.25.0
accelerate>=0.20.0
pandas>=1.3.0
numpy>=1.20.0
Pillow>=9.0.0
tqdm>=4.60.0
einops>=0.6.0
timm>=0.6.0 # For Swin Transformer
hpsv2>=0.1.0 # For HPSv2 evaluation (如果需要评估)
datasets>=2.10.0 # For downloading prompt dataset
tensorboard # For logging with accelerate

# 可能需要的其他依赖
sentencepiece
ftfy
protobuf # 如果遇到 protobuf 相关问题
```

## 使用流程

### 1. 提取 Prompts (`scripts/extract_prompts.py`)

从 Hugging Face Hub 下载数据集（如 Pick-a-Pic）并提取 prompts。

**命令:**
```bash
python scripts/extract_prompts.py \
	--dataset_name yuvalkirstain/pickapic_v1 \
	--split train \
	--caption_column caption \
	--output_dir ./data \
	--output_filename pickapic_prompts.txt
```
* 这会将 Pick-a-Pic 训练集的 `caption` 列保存到 `./data/pickapic_prompts.txt`。

### 2. 生成噪声序列数据集 (`src/generate_npd.py`)

使用步骤 1 的 prompts 文件和基础扩散模型（如 SDXL）生成噪声演化序列。

**命令:**
```bash
python src/generate_npd.py \
	--prompt_file ./data/pickapic_prompts.txt \
	--output_dir ./data/npd_sequence_dataset_sdxl/ \
	--num_steps 10 `# 生成 10 步的序列` \
	--max_prompts 1000 `# (可选) 只处理前 1000 个 prompt` \
	--output_size 1024
```
* `--num_steps`: 指定要生成的黄金噪声演化步数 (n)。
* `--max_prompts`: (可选) 限制处理的 prompt 数量，用于快速测试或生成小数据集。
* **重要:** 此过程非常耗时！可以考虑使用 `--max_prompts` 限制数量。输出保存在 `--output_dir` 指定的目录。

### 3. 训练模型

根据你想训练的模型选择对应的脚本：

**a) 训练原始 NPNet (`src/train.py`)**
* **注意:** 这个脚本需要 `(source_noise, target_noise)` 对的数据集，而不是序列数据集。你需要运行原始版本的 `generate_npd.py`（只生成一步 golden noise 并进行 HPSv2 筛选）来创建适配的数据。
* **命令示例:**
	```bash
	accelerate launch src/train.py \
		--dataset_dir <path_to_original_npd_data> \
		--output_dir ./npnet_training_output/ \
		--npnet_model_id SDXL \
		--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
		--num_epochs 30 \
		--batch_size 16 \
		# ... 其他训练参数 ...
		# --- NPNet 配置参数 ---
		# --svd_dropout --nt_adapter --cross_attention
	```

**b) 训练 Transformer 序列模型 (`src/train_seq_model.py`)**
* 使用步骤 2 生成的序列数据集。
* **命令示例:**
	```bash
	accelerate launch src/train_seq_model.py \
		--dataset_dir ./data/npd_sequence_dataset_sdxl/ \
		--output_dir ./seq_model_training_output/ \
		--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
		--npnet_model_id SDXL \
		--text_embed_dim 1024 `# 调整!` \
		--noise_resolution 128 \
		--patch_size 16 \
		--d_model 768 \
		--nhead 12 \
		--num_encoder_layers 4 \
		--num_decoder_layers 4 \
		--dim_feedforward 2048 \
		--max_seq_len 11 `# dataset_num_steps + 1` \
		--predict_variance `# 可选` \
		--num_epochs 50 \
		--batch_size 8 \
		--gradient_accumulation_steps 4 \
		--learning_rate 5e-5
		# ... 其他训练参数 ...
	```
* `--max_seq_len` 应设为数据集中 `num_steps + 1`（因为输入包含 $x_T$）。
* `--text_embed_dim` 需要根据你使用的 `base_model_id` 和 `encode_text` 函数的实际输出维度设置。

**c) 训练改进版 RNN 序列模型 (`src/train_rnn_model_v2.py`)**
* 使用步骤 2 生成的序列数据集。
* **命令示例:**
	```bash
	accelerate launch src/train_rnn_model_v2.py \
		--dataset_dir ./data/npd_sequence_dataset_sdxl/ \
		--output_dir ./rnn_v2_seq_model_output/ \
		--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
		--npnet_model_id SDXL \
		--text_embed_dim 1280 `# 调整! (SDXL pooled)` \
		--noise_resolution 128 \
		--cnn_base_filters 64 \
		--cnn_num_blocks 2 2 2 2 \
		--cnn_feat_dim 512 \
		--gru_hidden_size 1024 \
		--gru_num_layers 2 \
		--predict_variance `# 可选` \
		--predict_residual `# 可选` \
		--kl_weight 0.01 `# 可选 (如果 predict_variance=True)` \
		--num_epochs 50 \
		--batch_size 16 \
		--gradient_accumulation_steps 2 \
		--learning_rate 1e-4
		# ... 其他训练参数 ...
	```
* 调整 CNN 和 GRU 的参数。
* `--text_embed_dim` 同样需要确认，这里假设使用 SDXL 的 pooled embedding (1280)。
* 使用 `--predict_residual` 和 `--kl_weight` 启用相应功能。

### 4. 推理 (`src/inference.py`)

* **当前版本主要适配原始 NPNet 模型。**
* **命令示例 (使用训练好的 NPNet):**
	```bash
	python src/inference.py \
		--npnet_weights_path ./npnet_training_output/npnet_final.pth \
		--prompt "A cute cat wearing a wizard hat" \
		--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
		--output_dir ./inference_output/ \
		--npnet_model_id SDXL \
		--seed 123 \
		--generate_standard \
		# --- 添加与训练时匹配的 NPNet 配置 flags ---
		# --svd_dropout --nt_adapter --cross_attention
	```
* **对于序列模型 (Transformer/RNN):**
	* 你需要加载对应的序列模型权重。
	* 调用模型的 `generate_sequence(initial_noise, text_embed, num_steps)` 方法来生成噪声序列。
	* 选择序列中的最后一步噪声 (`generated_sequence[:, -1]`) 作为最终的黄金噪声。
	* 将这个最终噪声传递给基础扩散模型 (`pipe(..., latents=final_golden_noise, ...)`).
	* 这需要修改 `inference.py` 或创建一个新的推理脚本。

### 5. 评估 (`src/evaluate.py`)

* **当前版本主要适配原始 NPNet 模型。**
* **命令示例 (评估 NPNet):**
	```bash
	python src/evaluate.py \
		--npnet_weights_path ./npnet_training_output/npnet_final.pth \
		--evaluation_prompts_file <path/to/eval_prompts.txt> \
		--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
		--output_dir ./evaluation_output/ \
		--npnet_model_id SDXL \
		--metrics hpsv2 \
		--save_images \
		# --- 添加与训练时匹配的 NPNet 配置 flags ---
		# --svd_dropout --nt_adapter --cross_attention
	```
* **对于序列模型:**
	* 评估逻辑需要修改，以使用序列模型的 `generate_sequence` 方法获取最终黄金噪声，然后再进行图像生成和指标计算。
	* 可能需要创建一个新的评估脚本。

## 模型配置参数

在运行 `train.py`, `inference.py`, `evaluate.py` 时，需要使用与模型训练时**完全一致**的配置参数来确保模型结构匹配权重文件：

* `--svd_dropout`: (NPNet) SVDNoiseUnet 是否启用 dropout/layernorm。
* `--nt_adapter`: (NPNet) NoiseTransformer 是否启用 adapter。
* `--nt_finetune`: (NPNet) NoiseTransformer 是否启用 finetuning。
* `--nt_dropout`: (NPNet) NoiseTransformer 是否启用 dropout。
* `--cross_attention`: (NPNet) NPNet 是否启用 cross-attention。
* (对于序列模型，需要传递对应的 Transformer 或 RNN/CNN 参数)。

<!-- ## 引用

如果你的研究或工作得益于原始论文，请考虑引用：

```bibtex
@article{zhou2024golden,
	title={Golden Noise for Diffusion Models: A Learning Framework},
	author={Zhou, Zikai and Shao, Shitong and Bai, Lichen and Xu, Zhiqiang and Han, Bo and Xie, Zeke},
	journal={arXiv preprint arXiv:2411.09502},
	year={2024}
}
``` -->

<!-- ## 许可证

(在此处添加你的项目许可证信息, 例如 MIT License)

``` -->
