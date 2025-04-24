# scripts/evaluate_hps.py
import torch
import numpy as np
from PIL import Image
import os
import argparse
from tqdm.auto import tqdm
import pandas as pd
import traceback

# HPSv2 import
try:
    import hpsv2
    HPSV2_AVAILABLE = True
except ImportError:
    print("Error: hpsv2 library not found. Cannot perform evaluation.")
    print("Please install using: pip install hpsv2")
    HPSV2_AVAILABLE = False

def main(args):
    if not HPSV2_AVAILABLE:
        exit(1)

    print(f"Loading prompts from: {args.prompt_file}")
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts.")
        if not prompts: print("Error: No prompts found."); exit(1)
    except Exception as e: print(f"Error reading prompts file: {e}"); exit(1)

    # Define image directories
    std_img_dir = os.path.join(args.image_base_dir, "standard_output")
    gns_img_dir = os.path.join(args.image_base_dir, "gnsnet_output") # Matching batch inference output
    print(f"Reading standard images from: {std_img_dir}")
    print(f"Reading golden noise images from: {gns_img_dir}")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "hpsv2_evaluation.csv")
    print(f"Results will be saved to: {results_path}")

    evaluation_results = []
    not_found_count = 0

    for idx, prompt in enumerate(tqdm(prompts, desc="Evaluating HPSv2")):
        std_img_path = os.path.join(std_img_dir, f"{idx}.png")
        gns_img_path = os.path.join(gns_img_dir, f"{idx}.png")

        image_std = None
        image_gns = None
        score_std = np.nan
        score_gns = np.nan

        # Load images
        try:
            if os.path.exists(std_img_path):
                image_std = Image.open(std_img_path).convert("RGB")
            else:
                # print(f"Warning: Standard image not found for index {idx}: {std_img_path}")
                not_found_count += 1

            if os.path.exists(gns_img_path):
                image_gns = Image.open(gns_img_path).convert("RGB")
            else:
                # print(f"Warning: Golden noise image not found for index {idx}: {gns_img_path}")
                not_found_count += 1 # Count missing pairs

        except Exception as e:
            print(f"Error loading images for index {idx}: {e}")
            continue # Skip scoring if images can't be loaded

        # Calculate scores if both images were loaded
        if image_std is not None and image_gns is not None:
            try:
                # hpsv2.score returns a list, get the first element
                score_std = hpsv2.score(image_std, prompt, hps_version=args.hps_version)[0]
                score_gns = hpsv2.score(image_gns, prompt, hps_version=args.hps_version)[0]
            except Exception as e:
                print(f"Error calculating HPSv2 score for index {idx}: {e}")
                traceback.print_exc() # Show detailed error
                score_std = np.nan # Mark as NaN if scoring fails
                score_gns = np.nan
        elif image_std is not None: # Score standard only if golden is missing
                try: score_std = hpsv2.score(image_std, prompt, hps_version=args.hps_version)[0]
                except Exception as e: print(f"Error scoring standard image for index {idx}: {e}"); score_std = np.nan
        elif image_gns is not None: # Score golden only if standard is missing
                try: score_gns = hpsv2.score(image_gns, prompt, hps_version=args.hps_version)[0]
                except Exception as e: print(f"Error scoring golden image for index {idx}: {e}"); score_gns = np.nan


        evaluation_results.append({
            "index": idx,
            "prompt": prompt,
            "hpsv2_standard": score_std,
            "hpsv2_golden": score_gns,
            "hpsv2_diff": score_gns - score_std if not (np.isnan(score_std) or np.isnan(score_gns)) else np.nan
        })

    if not_found_count > 0:
            print(f"Warning: {not_found_count} image pairs were incomplete or missing.")

    # Save results to CSV
    results_df = pd.DataFrame(evaluation_results)
    try:
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved detailed evaluation results to: {results_path}")
    except Exception as e:
        print(f"Error saving results CSV: {e}")

    # Calculate and print average scores (ignoring NaNs)
    print("\n--- Evaluation Summary ---")
    print(f"Evaluated {len(results_df)} prompts.")
    avg_std = results_df['hpsv2_standard'].mean()
    avg_golden = results_df['hpsv2_golden'].mean()
    avg_diff = results_df['hpsv2_diff'].mean()
    # Count how many times golden > standard
    improvement_count = (results_df['hpsv2_diff'] > 0).sum()
    valid_comparisons = results_df['hpsv2_diff'].notna().sum()

    print(f"Average HPSv2:")
    print(f"  - Standard: {avg_std:.4f}")
    print(f"  - Golden:   {avg_golden:.4f}")
    print(f"  - Avg Diff (Golden - Standard): {avg_diff:.4f}")
    if valid_comparisons > 0:
            print(f"  - Golden > Standard: {improvement_count}/{valid_comparisons} ({improvement_count/valid_comparisons:.2%})")
    print("-" * 26)
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated images using HPSv2")
    parser.add_argument("--prompt_file", type=str, default="./data/pickapic_prompts.txt", help="Path to the text file containing prompts used for generation.")
    parser.add_argument("--image_base_dir", type=str, default="./inference_output/", help="Base directory where 'standard_output' and 'gnsnet_output' subdirectories reside.")
    parser.add_argument("--results_dir", type=str, default="./results/", help="Directory to save the evaluation results CSV.")
    parser.add_argument("--hps_version", type=str, default="v2.1", choices=["v2.0", "v2.1"], help="Version of HPSv2 model to use.")

    args = parser.parse_args()
    main(args)

