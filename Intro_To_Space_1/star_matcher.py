import os
import random
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
from ransac_line_fit import ransac_line_fit
from display_imgs import show_data

# === Helper Functions ===
def triangle_side_ratios(tri):
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    a = dist(tri[0], tri[1])
    b = dist(tri[1], tri[2])
    c = dist(tri[2], tri[0])
    sides = sorted([a, b, c])
    return [sides[1]/sides[0], sides[2]/sides[1]]  # ratio1, ratio2

def load_stars_from_txt(path, with_ids=False):
    """
    Load stars from a .txt file (with or without IDs).
    """
    stars = []
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if with_ids and len(parts) == 5:
                try:
                    id_ = int(parts[0].strip())
                    x = int(parts[1].strip())
                    y = int(parts[2].strip())
                    r = int(parts[3].strip())
                    b = int(parts[4].strip())
                    stars.append((id_, x, y, r, b))
                except:
                    continue
            elif not with_ids and len(parts) == 4:
                try:
                    x = int(parts[0].strip())
                    y = int(parts[1].strip())
                    r = int(parts[2].strip())
                    b = int(parts[3].strip())
                    stars.append((x, y, r, b))
                except:
                    continue
    return stars

def compute_affine_transform(src_pts, dst_pts):
    """
    Given 3 points from each image, compute the affine transformation matrix.
    """
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    transform = cv2.getAffineTransform(src_pts, dst_pts)

    def apply_transform(point):
        pt = np.dot(transform, np.array([point[0], point[1], 1]))
        return int(pt[0]), int(pt[1])

    return apply_transform

def find_closest_star(stars, target_point, max_distance=20):
    """
    Find the closest star to the given point based only on x,y distance.
    """
    best_match = None
    best_dist = float("inf")
    for star in stars:
        x = star[1] if len(star) == 5 else star[0]
        y = star[2] if len(star) == 5 else star[1]
        dist = np.sqrt((x - target_point[0])**2 + (y - target_point[1])**2)
        if dist < best_dist and dist <= max_distance:
            best_dist = dist
            best_match = star
    return best_match

def count_inliers(stars1, stars2, transform_fn, max_distance=20):
    """
    Count how many stars from stars1 are correctly mapped to stars2 within max_distance.
    """
    count = 0
    matches = {}

    for star in stars1:
        x = star[1] if len(star) == 5 else star[0]
        y = star[2] if len(star) == 5 else star[1]
        mapped_point = transform_fn((x, y))
        matched_star = find_closest_star(stars2, mapped_point, max_distance)
        matches[matched_star] = matches.get(matched_star, 0) + 1
        if matched_star is not None and matches[matched_star] <= 1:
            count += 1

    return count

def compute_matching_ratio(stars1, stars2, inlier_count):
    return inlier_count / min(len(stars1), len(stars2))

# === Main Matching Function ===

def match_stars(stars1, stars2, iterations=10000, max_distance=20, min_inliers_threshold=3, min_match_ratio=0.3):
    """
    Robust star matching using affine transformation and inlier verification.
    Matches based on (x, y) locations only, with filtering and validation.
    """
    if len(stars1) < 3 or len(stars2) < 3:
        raise ValueError("Not enough stars for matching.")

    # === Filter top 50 brightest stars ===
    stars1 = sorted(stars1, key=lambda s: s[4] if len(s) == 5 else 0, reverse=True)[:50]
    stars2 = sorted(stars2, key=lambda s: s[4] if len(s) == 5 else 0, reverse=True)[:50]

    # === Fit dominant lines via RANSAC ===
    line1, line_pts1 = ransac_line_fit(stars1)
    line2, line_pts2 = ransac_line_fit(stars2)

    if len(line_pts1) > 16:
        line_pts1 = random.sample(line_pts1, 16)
    if len(line_pts2) > 16:
        line_pts2 = random.sample(line_pts2, 16)

    best_transform = None
    best_inliers = -1
    best_s1 = None
    best_s2 = None
    tried = set()

    for _ in range(iterations):
        if len(line_pts1) < 3 or len(line_pts2) < 3:
            break

        s1 = tuple(random.sample(line_pts1, 3))
        s2 = tuple(random.sample(line_pts2, 3))

        key = (frozenset(s1), frozenset(s2))
        if key in tried:
            continue
        tried.add(key)

        src_pts = [(p[1], p[2]) if len(p) == 5 else (p[0], p[1]) for p in s1]
        dst_pts = [(p[1], p[2]) if len(p) == 5 else (p[0], p[1]) for p in s2]

        # Triangle consistency check
        ratios1 = triangle_side_ratios(src_pts)
        ratios2 = triangle_side_ratios(dst_pts)
        if any(abs(r1 - r2) > 0.25 for r1, r2 in zip(ratios1, ratios2)):
            continue

        transform_fn = compute_affine_transform(src_pts, dst_pts)
        inliers = count_inliers(line_pts1, line_pts2, transform_fn, max_distance=max_distance)

        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = transform_fn
            best_s1 = s1
            best_s2 = s2

        if best_inliers >= min(len(line_pts1), len(line_pts2)) * 0.9:
            break

    # Accept best match even if under threshold (but warn)
    if best_transform is None:
        print("No valid transformation found.")
        return [], [], [], line1, line_pts1, line2, line_pts2, 0.0

    match_ratio = compute_matching_ratio(line_pts1, line_pts2, best_inliers)

    if best_inliers < min_inliers_threshold or match_ratio < min_match_ratio:
        print(f"⚠️ Weak match accepted: {best_inliers} inliers, ratio = {match_ratio:.2f}")

    mapped_stars = []
    for star in line_pts1:
        x = star[1] if len(star) == 5 else star[0]
        y = star[2] if len(star) == 5 else star[1]
        mapped = best_transform((x, y))
        mapped_stars.append([(x, y), mapped])

    return mapped_stars, best_s1, best_s2, line1, line_pts1, line2, line_pts2, match_ratio

def save_matches(filepath, mapped_stars, image_size, match_ratio=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("source , destination\n")
        for src, dst in mapped_stars:
            if 0 <= dst[0] < image_size[0] and 0 <= dst[1] < image_size[1]:
                f.write(f"{src} , {dst}\n")
            else:
                f.write(f"{src} , no matching\n")
        if match_ratio is not None:
            f.write(f"\nMatching Ratio: {match_ratio:.4f}")

# === Main Execution (Optional for test) ===

if __name__ == "__main__":
    register_heif_opener()

    size = (600, 600)
    stars1_path = './star_detection_results/stars_ST_db1.txt'
    stars2_path = './star_detection_results/stars_ST_db2.txt'
    image1_path = './imgs/ST_db1.png'
    image2_path = './imgs/ST_db2.png'
    output_path = './star_matching_results/matches.txt'

    stars1 = load_stars_from_txt(stars1_path, with_ids=True)
    stars2 = load_stars_from_txt(stars2_path, with_ids=True)

    mapped_stars, src_pts, dst_pts, line1, pts1, line2, pts2, ratio = match_stars(stars1, stars2)

    save_matches(output_path, mapped_stars, size, ratio)

    img1 = np.array(Image.open(image1_path).resize(size).convert("RGB"))
    img2 = np.array(Image.open(image2_path).resize(size).convert("RGB"))

    show_data(src_pts, dst_pts, pts1, line1, pts2, line2, mapped_stars, img1, img2)

    print(f"Matching completed! Ratio: {ratio:.4f}")