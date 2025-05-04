import os
import numpy as np
from star_matcher import load_stars_from_txt, match_stars, save_matches
from display_imgs import show_data
from PIL import Image

# === Settings ===
size = (600, 600)

# Matching combinations
pairs = [
    {
        "name": "fr1_to_fr1",
        "img1": "fr1.jpg",
        "img2": "fr1.jpg",
        "stars1": "stars_fr1.txt",
        "stars2": "stars_fr1.txt"
    },
    {
        "name": "fr2_to_fr2",
        "img1": "fr2.jpg",
        "img2": "fr2.jpg",
        "stars1": "stars_fr2.txt",
        "stars2": "stars_fr2.txt"
    },
    {
        "name": "ST_db1_to_ST_db1",
        "img1": "ST_db1.png",
        "img2": "ST_db1.png",
        "stars1": "stars_ST_db1.txt",
        "stars2": "stars_ST_db1.txt"
    },
    {
        "name": "ST_db2_to_ST_db2",
        "img1": "ST_db2.png",
        "img2": "ST_db2.png",
        "stars1": "stars_ST_db2.txt",
        "stars2": "stars_ST_db2.txt"
    },
    {
        "name": "fr1_to_ST_db1",
        "img1": "fr1.jpg",
        "img2": "ST_db1.png",
        "stars1": "stars_fr1.txt",
        "stars2": "stars_ST_db1.txt"
    },
    {
        "name": "fr2_to_ST_db2",
        "img1": "fr2.jpg",
        "img2": "ST_db2.png",
        "stars1": "stars_fr2.txt",
        "stars2": "stars_ST_db2.txt"
    },
    {
        "name": "fr2_to_ST_db1",
        "img1": "fr2.jpg",
        "img2": "ST_db1.png",
        "stars1": "stars_fr2.txt",
        "stars2": "stars_ST_db1.txt"
    },
    {
        "name": "fr1_to_ST_db2",
        "img1": "fr1.jpg",
        "img2": "ST_db2.png",
        "stars1": "stars_fr2.txt",
        "stars2": "stars_ST_db2.txt"
    },
    {
        "name": "fr1_to_fr2",
        "img1": "fr1.jpg",
        "img2": "fr2.jpg",
        "stars1": "stars_fr1.txt",
        "stars2": "stars_fr2.txt"
    },
    {
        "name": "fr2_to_fr1",
        "img1": "fr2.jpg",
        "img2": "fr1.jpg",
        "stars1": "stars_fr2.txt",
        "stars2": "stars_fr1.txt"
    }
]

# Folder setup
input_folder = "./imgs"
stars_folder = "./star_detection_results"
output_folder = "./star_matching_results"
os.makedirs(output_folder, exist_ok=True)

# === Loop over all pairs ===
for pair in pairs:
    print(f"\n Matching: {pair['stars1']} <-> {pair['stars2']}")

    # File paths
    image1_path = os.path.join(input_folder, pair["img1"])
    image2_path = os.path.join(input_folder, pair["img2"])

    stars1_path = os.path.join(stars_folder, pair["stars1"])
    stars2_path = os.path.join(stars_folder, pair["stars2"])

    output_match_path = os.path.join(output_folder, f"matches_{pair['name']}.txt")

    # Load stars with IDs
    stars1 = load_stars_from_txt(stars1_path, with_ids=True)
    stars2 = load_stars_from_txt(stars2_path, with_ids=True)

    # Sort stars to ensure consistent ordering (important for self-match cases)
    stars1 = sorted(stars1, key=lambda s: (s[1], s[2]))  # Sort by x, y
    stars2 = sorted(stars2, key=lambda s: (s[1], s[2]))

    if len(stars1) < 2 or len(stars2) < 2:
        print(f" Skipping: Not enough stars in {pair['name']} (s1={len(stars1)}, s2={len(stars2)})")
        continue

    # Match
    mapped_stars, src_pts, dst_pts, line1, pts1, line2, pts2, ratio = match_stars(stars1, stars2)

    # Save results
    save_matches(output_match_path, mapped_stars, size, match_ratio=ratio)

    # Load and convert images to RGB
    img1 = np.array(Image.open(image1_path).resize(size).convert("RGB"))
    img2 = np.array(Image.open(image2_path).resize(size).convert("RGB"))

    # Show results
    show_data(src_pts, dst_pts, pts1, line1, pts2, line2, mapped_stars, img1, img2)

    # Report
    print(f" Match {pair['name']} completed")
    print(f" Matching ratio: {ratio:.4f}")
    print(f" Results saved to: {output_match_path}")