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

# Run detection
python run_detection.py

# Run matching
python run_matching.py

# 🔍 Matching Output
Each match file (e.g., matches_fr1_to_fr2.txt) contains:

A list of matched star coordinates from source to destination.

A computed matching ratio showing the proportion of valid matches.

 # Evaluation & Results
**Star matching results:**
![fr1_to_fr1](https://github.com/user-attachments/assets/eaf94daf-e86d-4e6e-956a-c0d1f4b1131a)
![‏‏צילום מסך (1147)](https://github.com/user-attachments/assets/1b962db2-4bc6-419f-960f-17ffe95224c3)

![‏‏צילום מסך (1151)](https://github.com/user-attachments/assets/1cd6f9d5-deeb-4982-8bc1-1808c61ebbd2)

**Detection results:**

![‏‏צילום מסך (1155)](https://github.com/user-attachments/assets/273f9ffd-f3a1-4a6e-8ae1-1f13e2e3d9be)
![‏‏צילום מסך (1154)](https://github.com/user-attachments/assets/b020c11f-7f20-4fc1-b698-c5d7e9aa2826)


This project was developed by a Aseel Ahmad as part of the Introduction to Space Engineering course
