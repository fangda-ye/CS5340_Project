# Golden Noise for Diffusion Models - Implementation

This project implements the concepts presented in the paper "Golden Noise for Diffusion Models: A Learning Framework" (arXiv:2411.09502). It provides tools to generate a Noise Prompt Dataset (NPD), train a Noise Prompt Network (NPNet), and use the trained NPNet to potentially improve image generation quality from text-to-image diffusion models like Stable Diffusion XL.

## Project Structure

```
CS5340_PROJECT/
├── data/                     # Directory for datasets (e.g., generated NPD)
│   └── npd_dataset_sdxl/     # Example dataset directory
│       ├── noises/           # Saved noise tensors (.pt files)
│       └── metadata.csv      # Metadata linking prompts and noise files
├── doc/                      # Documentation or paper related files (optional)
├── model/                    # Model definitions
│   ├── __init__.py
│   ├── Attention.py
│   ├── NoiseTransformer.py
│   ├── SVDNoiseUnet.py
│   └── npnet.py              # Core NPNet model definition
├── src/                      # Source code for scripts
│   ├── generate_npd.py       # Script to generate the Noise Prompt Dataset
│   ├── dataset.py            # PyTorch Dataset and DataLoader for NPD
│   ├── train.py              # Script to train the NPNet model
│   ├── inference.py          # Script to generate images using trained NPNet
│   └── evaluate.py           # Script to evaluate NPNet performance
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Prerequisites:**
	* Python 3.8+
	* PyTorch (tested with 1.13+, CUDA recommended)
	* CUDA Toolkit (if using GPU)

2.  **Clone the repository:**
	```bash
	git clone <your-repo-url>
	cd CS5340_PROJECT
	```

3.  **Install Dependencies:**
	It's highly recommended to use a virtual environment (like `conda` or `venv`).
	```bash
	pip install -r requirements.txt
	```
	*(You'll need to create the `requirements.txt` file, see below)*

## `requirements.txt`

Create a file named `requirements.txt` in the project root with the following content (adjust versions as needed):

```txt
torch>=1.13.0
torchvision
torchaudio
diffusers>=0.21.0 # Or a version compatible with SDXL/HunyuanDiT used
transformers>=4.25.0 # For CLIP/text encoders and HPSv2
accelerate>=0.20.0 # For simplified training script
pandas>=1.3.0
numpy>=1.20.0
Pillow>=9.0.0
tqdm>=4.60.0
einops>=0.6.0
timm>=0.6.0 # For Swin Transformer in NoiseTransformer
hpsv2>=0.1.0 # For HPSv2 evaluation
# Add other metric libraries if you implement them:
# clip-score
# ImageReward
sentencepiece # Often needed by tokenizers
ftfy # Often needed by CLIP preprocessor
```
Install using `pip install -r requirements.txt`.

## Usage

### 1. Data Generation (`generate_npd.py`)

This script generates the Noise Prompt Dataset (NPD) using a base diffusion model (e.g., SDXL) and filters pairs using HPSv2.

**Command:**
```bash
python src/generate_npd.py \
	--prompt_file ./data/prompts.txt \
	--output_dir ./data/npd_dataset_sdxl/ \
	--max_samples 10 # (Optional) Limit number of selected pairs
	# Add other arguments as needed (e.g., --output_size)
```

* `--prompt_file`: Path to a text file (one prompt per line) or CSV file (specify column with `--prompt_column`).
* `--output_dir`: Where to save the generated `metadata.csv` and `noises/` directory.
* `--max_samples`: Stops generation after collecting this many *valid* pairs (useful for testing). Default processes all prompts.
* **Note:** This process is computationally expensive. Using a smaller base model (like SD 1.5) for generation might be faster if SDXL is too slow, but the paper used SDXL. Ensure `hpsv2` is installed for filtering.

### 2. Training (`train.py`)

This script trains the NPNet model using the generated NPD dataset and `accelerate` for handling distribution and mixed precision.

**Command Example (Single GPU):**
```bash
accelerate launch src/train.py \
	--dataset_dir ./data/npd_dataset_sdxl/ \
	--output_dir ./npnet_training_output/ \
	--npnet_model_id SDXL \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--num_epochs 30 \
	--batch_size 16 \
	--learning_rate 1e-4 \
	--mixed_precision fp16 \
	--gradient_accumulation_steps 1 \
	--log_steps 50 \
	--save_steps 500 \
	# --- Add NPNet config flags used during training ---
	# --svd_dropout
	# --nt_adapter
	# --nt_finetune
	# --nt_dropout
	# --cross_attention
	# --- Optional ---
	# --resume_from <path/to/checkpoint.pth>
```

* Adjust `--batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--num_epochs` based on your hardware and dataset size.
* Use the **same NPNet configuration flags** (`--svd_dropout`, `--nt_adapter`, etc.) that you intend to use for inference/evaluation.
* `accelerate` handles device placement. For multi-GPU, configure `accelerate` first (`accelerate config`) and then run the same command.

### 3. Inference (`inference.py`)

Generates images using a trained NPNet model and a base diffusion pipeline.

**Command:**
```bash
python src/inference.py \
	--npnet_weights_path ./npnet_training_output/npnet_final.pth \
	--prompt "A high-resolution photo of a golden retriever puppy playing in a field of flowers" \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--output_dir ./npnet_output_images/ \
	--npnet_model_id SDXL \
	--seed 42 \
	--num_inference_steps 30 \
	--guidance_scale 5.5 \
	--generate_standard \
	# --- Add NPNet config flags matching the trained model ---
	# --svd_dropout
	# --nt_adapter
	# --nt_finetune
	# --nt_dropout
	# --cross_attention
```
* `--npnet_weights_path`: Path to your trained `.pth` file.
* `--prompt`: The text prompt to generate.
* `--base_model_id`: The diffusion model pipeline to use for generation.
* Make sure the NPNet config flags match the model specified in `--npnet_weights_path`.
* `--generate_standard`: Optionally generates an image using the original noise for comparison.

### 4. Evaluation (`evaluate.py`)

Evaluates a trained NPNet by generating images for multiple prompts and calculating metrics.

**Command:**
```bash
python src/evaluate.py \
	--npnet_weights_path ./npnet_training_output/npnet_final.pth \
	--evaluation_prompts_file <path/to/eval_prompts.txt_or_csv> \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--output_dir ./npnet_evaluation_output/ \
	--npnet_model_id SDXL \
	--metrics hpsv2 \
	--hps_version v2.1 \
	--seed 1000 \
	--save_images \
	# --- Add NPNet config flags matching the trained model ---
	# --svd_dropout
	# --nt_adapter
	# --nt_finetune
	# --nt_dropout
	# --cross_attention
```
* `--evaluation_prompts_file`: File containing prompts for evaluation.
* `--metrics`: List of metrics to compute (currently supports `hpsv2`).
* `--save_images`: Optionally save the generated standard and golden images for inspection.
* Results (including average scores and a detailed CSV) will be saved in `--output_dir`.

## Model Configuration Flags

When training, inferring, or evaluating, ensure you use consistent flags for the NPNet configuration:

* `--svd_dropout`: Use if `SVDNoiseUnet` was trained with `enable_drop=True`.
* `--nt_adapter`: Use if `NoiseTransformer` was trained with `enable_adapter=True`.
* `--nt_finetune`: Use if `NoiseTransformer` was trained with `enable_finetune=True`.
* `--nt_dropout`: Use if `NoiseTransformer` was trained with `enable_dropout=True`.
* `--cross_attention`: Use if `NPNet` was trained with `enable_cross_attention=True`.

These flags ensure the loaded model structure matches the saved weights.

## Citation

If you use concepts or code related to the original paper, please cite:

```bibtex
@article{zhou2024golden,
	title={Golden Noise for Diffusion Models: A Learning Framework},
	author={Zhou, Zikai and Shao, Shitong and Bai, Lichen and Xu, Zhiqiang and Han, Bo and Xie, Zeke},
	journal={arXiv preprint arXiv:2411.09502},
	year={2024}
}
```

*(Please replace the above citation with the final published version if available)*

## License

*(Specify your project's license here, e.g., MIT License)*
```
