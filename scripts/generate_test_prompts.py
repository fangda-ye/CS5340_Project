# scripts/generate_test_prompts.py
import random
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"
source_file = data_dir / "prompts.txt"
target_file = data_dir / "test_prompts.txt"

with source_file.open("r", encoding="utf-8") as f:
    lines = f.readlines()

if len(lines) < 100:
    raise ValueError(f"Not enough lines in {source_file}, only found {len(lines)}")

sampled_lines = random.sample(lines, 100)

with target_file.open("w", encoding="utf-8") as f:
    f.writelines(sampled_lines)

print(f"Successfully wrote 100 random prompts to {target_file}")