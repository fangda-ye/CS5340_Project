# Sequential Golden Noise Generation with RNN

This project explores the concept of generating a sequence of "golden noises" for text-to-image diffusion models, inspired by the ideas in "Golden Noise for Diffusion Models: A Learning Framework" (arXiv:2411.09502). Instead of predicting a single optimal noise, this implementation focuses on modeling the *evolutionary process* where noise is iteratively refined based on a text prompt, using a Recurrent Neural Network (RNN) based architecture (`NoiseSequenceRNN_v3`).

The core workflow involves:
1.  Generating a dataset where each sample contains an initial noise ($x_T$) and a sequence of subsequent noises ($[x'_1, ..., x'_n]$) obtained through iterative DDIM Denoise/Inversion steps conditioned on a text prompt ($c$).
2.  Training an RNN model (`NoiseSequenceRNN_v3`) to learn the conditional transition $p_\theta(x'_k | x'_{k-1}, c)$, predicting the distribution of the next noise state.
3.  Using the trained RNN model during inference to generate a sequence of refined noises and using the final noise ($\hat{x}'_n$) as an improved starting point for a standard diffusion model (e.g., SDXL).

## Project Structure

```
CS5340_PROJECT/
├── data/                     # Datasets
│   ├── prompts.txt           # Input prompts for dataset generation
│   ├── test_prompts.txt      # Prompts for evaluation/testing
│   └── npd_sequence_dataset_sdxl/ # Generated sequence dataset (example)
│       ├── sequences/        # Saved noise sequence files (.pt)
│       └── metadata.csv      # Metadata linking prompts and sequence files
├── doc/                      # Documentation (optional)
├── inference_output/         # Default output directory for inference images
│   ├── standard_output/      # Images from standard noise
│   └── gnsnet_output/        # Images from golden noise (RNN output)
├── model/                    # Model definitions
│   ├── __init__.py
│   └── rnn_seq_model_v3.py   # RNN Sequence Model (V3) definition
├── output/                   # Default output directory for training checkpoints
│   └── rnn_v3_seq_model_output/ # Example training output dir
├── references/               # Reference papers (optional)
├── results/                  # Default output directory for evaluation results
├── scripts/                  # Utility and evaluation scripts
│   ├── batch_inference.py    # Generate images for multiple prompts
│   ├── evaluate_hps.py       # Evaluate generated images using HPSv2
│   ├── evalute.sh            # Example evaluation script (shell)
│   ├── extract_prompts.py    # Extract prompts from Hugging Face dataset
│   ├── generate_test_prompts.py # (Utility for creating test prompts)
│   ├── inference.sh          # Example inference script (shell)
│   └── train.sh              # Example training script (shell)
├── src/                      # Core source code
│   ├── dataset.py            # PyTorch Dataset and DataLoader for sequence data
│   ├── generate_npd_series.py # Script to generate the noise sequence dataset
│   ├── inference_rnn.py      # Script to run inference with the RNN model
│   └── train_rnn_model_v3.py # Script to train the RNN model (V3)
├── LICENSE                   # Project License
└── README.md                 # This file
```
*(Note: Original NPNet related files like `NoiseTransformer.py`, `SVDNoiseUnet.py`, `npnet.py`, `train.py` etc. are assumed removed as per user request)*

## Setup

1.  **Prerequisites:**
	* Python 3.8+
	* PyTorch (CUDA recommended)
	* CUDA Toolkit & compatible NVIDIA driver (if using GPU)

2.  **Clone Repository:**
	```bash
	git clone <your-repo-url>
	cd CS5340_PROJECT
	```

3.  **Create Environment & Install Dependencies:**
	Using Conda is recommended:
	```bash
	conda create -n golden_rnn python=3.8 -c conda-forge -y
	conda activate golden_rnn
	# Install PyTorch matching your CUDA version (check PyTorch website)
	# Example for CUDA 11.8:
	# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	# Example for CUDA 12.1:
	# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

	# Install other dependencies
	pip install -r requirements.txt
	```

## `requirements.txt` (Example)

```txt
# PyTorch (Install via Conda specific to your CUDA version first)
# torch>=1.13.0
# torchvision
# torchaudio

# Core Libraries
diffusers>=0.21.0
transformers>=4.25.0
accelerate>=0.20.0
pandas>=1.3.0
numpy>=1.20.0
Pillow>=9.0.0
tqdm>=4.60.0
einops>=0.6.0 # Might be needed by underlying models
timm>=0.6.0   # Might be needed by underlying models

# Specific Tools
hpsv2>=0.1.0 # For evaluation script
datasets>=2.10.0 # For extracting prompts
tensorboard # For logging with accelerate

# Optional/Common
sentencepiece
ftfy
protobuf
```
*(Ensure PyTorch is installed first via Conda matching your CUDA version, then run `pip install -r requirements.txt`)*

## Step-by-Step Usage

### Step 1: Prepare Prompts

Use the script to download prompts from a dataset like Pick-a-Pic.

```bash
python scripts/extract_prompts.py \
	--dataset_name yuvalkirstain/pickapic_v1 \
	--split train \
	--output_dir ./data \
	--output_filename prompts.txt
```
This will save prompts to `./data/prompts.txt`. You can also prepare your own `.txt` file with one prompt per line. Create a separate file (e.g., `./data/test_prompts.txt`) for evaluation later.

### Step 2: Generate Noise Sequence Dataset

Use the prepared prompts and a base diffusion model (SDXL recommended) to generate the sequential noise data.

```bash
# Ensure you are in the CS5340_PROJECT directory
python src/generate_npd_series.py \
	--prompt_file ./data/prompts.txt \
	--output_dir ./data/npd_sequence_dataset_sdxl/ \
	--num_steps 10 \
	--max_prompts 1000 # Adjust as needed for dataset size
```
* `--num_steps`: Number of golden noise steps ($n$) to generate per prompt.
* `--max_prompts`: Limits the number of prompts processed (useful for creating smaller test datasets). Remove to process all prompts.
* This script saves `_source.pt` and `_golden_sequence.pt` files in the `sequences` sub-directory and creates a `metadata.csv`.
* **Warning:** This step is computationally intensive and time-consuming.

### Step 3: Train the RNN Sequence Model

Train the `NoiseSequenceRNN_v3` model on the generated dataset using `accelerate`.

```bash
# Ensure PYTHONPATH includes the project root if running from root
# export PYTHONPATH=. (or use PYTHONPATH=. before accelerate)

accelerate launch src/train_rnn_model_v3.py \
	--dataset_dir ./data/npd_sequence_dataset_sdxl/ \
	--output_dir ./output/rnn_v3_seq_model_output/ \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--npnet_model_id SDXL `# For text dim hint` \
	--text_embed_dim 1280 `# Adjust if using different base model/embedding` \
	--noise_resolution 128 \
	--cnn_base_filters 64 \
	--cnn_num_blocks 2 2 2 2 \
	--cnn_feat_dim 512 \
	--gru_hidden_size 1024 \
	--gru_num_layers 2 \
	--predict_variance `# Add if you want variance prediction` \
	--kl_weight 0.01 `# Add if using variance prediction and KL loss` \
	--num_epochs 50 \
	--batch_size 8 `# Adjust based on GPU memory` \
	--gradient_accumulation_steps 4 `# Adjust based on GPU memory` \
	--learning_rate 1e-4 \
	--mixed_precision fp16 \
	--save_steps 1000 \
	--max_checkpoints 3 `# Limit disk usage`
```
* Adjust hyperparameters (batch size, learning rate, model dimensions, etc.) based on your resources and dataset.
* The `--text_embed_dim` should match the dimension of the text embedding used (e.g., 1280 for SDXL's pooled CLIP-G).
* Checkpoints and logs will be saved in `--output_dir`.

### Step 4: Inference (Single Prompt)

Use the trained RNN model to generate an image for a specific prompt.

```bash
# Ensure PYTHONPATH includes the project root if running from root
# export PYTHONPATH=.

python src/inference_rnn.py \
	--rnn_weights_path ./output/rnn_v3_seq_model_output/rnn_v3_model_final.pth \
	--prompt "A futuristic cityscape at sunset, synthwave style" \
	--output_dir ./inference_output/ \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--num_gen_steps 10 `# MUST match dataset num_steps` \
	--num_inference_steps 30 \
	--guidance_scale 5.5 \
	--seed 12345 \
	--generate_standard \
	--dtype float16 \
	# --- Add model config flags matching the trained model ---
	# e.g., --predict_variance (if trained with it)
	# (Other model dimension args are hardcoded in this script for now)
```
* Replace `--rnn_weights_path` with the actual path to your trained model.
* Set `--num_gen_steps` to the same value used during dataset generation.
* The script will save both the standard noise image and the golden noise image (if `--generate_standard` is used).

### Step 5: Batch Inference (Multiple Prompts)

Generate images for all prompts listed in a file.

```bash
# Ensure PYTHONPATH includes the project root if running from root
# export PYTHONPATH=.

python scripts/batch_inference.py \
	--prompt_file ./data/test_prompts.txt \
	--output_base_dir ./inference_output/ \
	--rnn_weights_path ./output/rnn_v3_seq_model_output/rnn_v3_model_final.pth \
	--base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
	--num_gen_steps 10 \
	--start_seed 1000 # Use a different starting seed than training/single inference
	# --- Add necessary model config flags ---
	# --predict_variance
```
* This script reads prompts from `--prompt_file`.
* It saves standard images to `<output_base_dir>/standard_output/` and golden noise images to `<output_base_dir>/gnsnet_output/`.
* Images are named `{index}.png` corresponding to the line number in the prompt file.

### Step 6: Evaluation (HPSv2)

Evaluate the generated image pairs using the HPSv2 score.

```bash
# Ensure hpsv2 is installed: pip install hpsv2
python scripts/evaluate_hps.py \
	--prompt_file ./data/test_prompts.txt \
	--image_base_dir ./inference_output/ \
	--results_dir ./results/ \
	--hps_version v2.1
```
* `--prompt_file` should be the same file used for batch inference.
* `--image_base_dir` points to the directory containing `standard_output` and `gnsnet_output`.
* Results are saved to `<results_dir>/hpsv2_evaluation.csv`.

<!-- ## Citation

If using concepts from the original paper, please cite:
```bibtex
@article{zhou2024golden,
	title={Golden Noise for Diffusion Models: A Learning Framework},
	author={Zhou, Zikai and Shao, Shitong and Bai, Lichen and Xu, Zhiqiang and Han, Bo and Xie, Zeke},
	journal={arXiv preprint arXiv:2411.09502},
	year={2024}
}
```

## License

(Specify your project's license here, e.g., MIT License)
``` -->
