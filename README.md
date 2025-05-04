# Star-Tracker
# ⭐ Star Detection and Matching Project

This project implements a robust pipeline for detecting stars in astronomical images and matching them across different frames using geometric transformations and brightness filtering.

## 📌 Overview

The project is divided into two main components:

1. **Star Detection**: Identifies and extracts features (coordinates, radius, brightness) of stars from input images.
2. **Star Matching**: Matches stars between two images using affine transformations computed from triangle sampling and RANSAC-based line detection.

---

## 🗂️ Project Structure

├── imgs/ # Input images (e.g., fr1.jpg, ST_db1.png)
├── star_detection_results/ # Output .txt files with detected stars
├── star_matching_results/ # Output .txt files with star match results
├── imgs_detected/ # Overlay images with detected stars
├── display_imgs.py # Visualization of matches
├── star_finder.py # Star detection logic
├── star_matcher.py # Matching algorithm
├── run_detection.py # Runs detection for a list of images
├── run_matching.py # Runs matching over selected image pairs


---

## ⚙️ How It Works

### Star Detection

- Run `run_detection.py` to detect stars in images.
- Outputs files like `stars_fr1.txt` that contain:  
  `id, x, y, radius, brightness`

### Star Matching

- Run `run_matching.py` to match stars between pairs of images.
- The algorithm:
  - Detects dominant lines using RANSAC.
  - Samples triangles from these lines.
  - Computes an affine transformation.
  - Maps stars and checks if they match based on spatial distance and brightness similarity.
- Outputs match files like `matches_fr1_to_ST_db1.txt`.

---

## ✅ Example Usage

bash
# Run detection
python run_detection.py

# Run matching
python run_matching.py

**🔍 Matching Output**
Each match file (e.g., matches_fr1_to_fr2.txt) contains:

A list of matched star coordinates from source to destination.

A computed matching ratio showing the proportion of valid matches.

 **Evaluation & Results**
Self-matching (e.g., fr1 to fr1) results in high accuracy.[fr1_to_fr1] (https://github.com/user-attachments/assets/730b8b6d-99fe-416d-a5ca-46c5babcdcce)
![‏‏fr2_to_ST_db2](https://github.com/user-attachments/assets/63566a8f-3d4b-42cf-9b76-29e84fbc122a)

Optional visualization saved as a side-by-side image with connecting lines.

