# scripts/extract_prompts.py
import argparse
import os
from tqdm.auto import tqdm

try:
    # datasets library is needed to download from Hugging Face Hub
    from datasets import load_dataset, DownloadConfig
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Please install it using: pip install datasets")
    exit(1)

def extract_and_save_prompts(
    dataset_name="yuvalkirstain/pickapic_v1", # Default Pick-a-Pic dataset
    split="train",                             # Use the training split as per paper
    caption_column="caption",                  # Column name containing prompts
    output_dir="../data",                      # Save to data directory relative to scripts/
    output_filename="pickapic_prompts.txt",
    cache_dir=None                             # Optional: specify cache directory for datasets
):
    """
    Downloads a dataset from Hugging Face Hub, extracts prompts from a specified
    column, and saves them to a text file.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        split (str): Dataset split to use (e.g., 'train').
        caption_column (str): Name of the column containing the prompts/captions.
        output_dir (str): Directory to save the output text file.
        output_filename (str): Name of the output text file.
        cache_dir (str, optional): Directory to cache downloaded datasets. Defaults to None.
    """
    print(f"Loading dataset '{dataset_name}', split '{split}'...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Configure download mode if needed (e.g., resume downloads)
        download_config = DownloadConfig(resume_download=True, cache_dir=cache_dir)
        # Load the dataset streamingly to avoid downloading everything if not needed
        # Note: Streaming might be slower for full extraction but saves disk space initially.
        # Use streaming=False if you prefer to download the whole dataset first.
        dataset = load_dataset(dataset_name, split=split, streaming=True, download_config=download_config)
        print("Dataset loaded successfully (streaming mode).")

    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print("Please check the dataset name and ensure you have internet access.")
        exit(1)

    print(f"Extracting prompts from column '{caption_column}'...")
    print(f"Saving prompts to: {output_path}")

    count = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Iterate through the dataset using streaming
            # Wrap with tqdm for progress (requires knowing the dataset size,
            # which streaming doesn't provide easily. We'll estimate or just show iteration count).
            # For non-streaming: progress_bar = tqdm(dataset, desc="Extracting prompts")
            progress_bar = tqdm(desc="Extracting prompts", unit=" prompts")
            for example in dataset:
                if caption_column in example:
                    prompt = example[caption_column]
                    if isinstance(prompt, str) and prompt.strip():
                        f_out.write(prompt.strip() + '\n')
                        count += 1
                        progress_bar.update(1)
                    # else:
                    #     print(f"Warning: Skipping invalid or empty prompt in example: {example}")
                # else:
                #     print(f"Warning: Caption column '{caption_column}' not found in example: {example}")
                #     print("Please verify the column name.")
                #     # Optional: break here if column is consistently missing
                #     # break
        progress_bar.close()
        print(f"\nSuccessfully extracted and saved {count} prompts to {output_path}")

    except KeyError:
            print(f"\nError: Column '{caption_column}' not found in the dataset.")
            print("Please check the dataset structure and provide the correct column name using --caption_column.")
            # Clean up potentially partially written file
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        print(f"\nError writing prompts to file: {e}")
        # Clean up potentially partially written file
        if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract prompts from a Hugging Face dataset.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yuvalkirstain/pickapic_v1",
        help="Name of the dataset on Hugging Face Hub (default: Pick-a-Pic v1)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="Name of the column containing prompts/captions (default: caption)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data", # Default relative path to save in project's data folder
        help="Directory to save the output prompts.txt file (default: ../data)."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="pickapic_prompts.txt",
        help="Name for the output text file (default: pickapic_prompts.txt)."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory to cache downloaded datasets."
    )

    args = parser.parse_args()

    # --- Execute Extraction ---
    extract_and_save_prompts(
        dataset_name=args.dataset_name,
        split=args.split,
        caption_column=args.caption_column,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        cache_dir=args.cache_dir
    )
