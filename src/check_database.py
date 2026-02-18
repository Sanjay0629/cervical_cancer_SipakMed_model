# Create check_dataset.py
import os
from pathlib import Path

data_dir = "data/raw"
classes = [
    "Superficial-Intermediate",
    "Parabasal", 
    "Metaplastic",
    "Dyskeratotic",
    "Koilocytotic"
]

print("\n" + "="*60)
print("DATASET CLASS DISTRIBUTION")
print("="*60)

total = 0
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    num_images = len(list(Path(class_dir).glob("*.bmp")))
    total += num_images
    print(f"{class_name:30s}: {num_images:5d} images")

print("-"*60)
print(f"{'TOTAL':30s}: {total:5d} images")
print("="*60)

# Check for severe imbalance
counts = []
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    counts.append(len(list(Path(class_dir).glob("*.bmp"))))

max_count = max(counts)
min_count = min(counts)
imbalance_ratio = max_count / min_count

print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print("⚠️  WARNING: Severe class imbalance detected!")
    print("   Some classes have 3x more images than others")
else:
    print("✅ Dataset is reasonably balanced")