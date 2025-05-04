import os
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIC format support
register_heif_opener()

def detect_stars(image: np.ndarray, img_size: tuple, max_stars: int = 1000) -> list:
    """
    Detects stars in a grayscale image using Hough Circle Transform.

    Args:
        image (np.ndarray): Grayscale input image.
        img_size (tuple): Image size (width, height).
        max_stars (int): Maximum number of stars to detect.

    Returns:
        list: List of tuples (x, y, r, b) for each detected star.
    """
    circles = cv2.HoughCircles(
        image,
        method=cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=20,
        param1=250,
        param2=2,
        minRadius=2,
        maxRadius=6
    )

    if circles is None:
        return []

    circles = np.uint16(np.around(circles))

    # Reduce stars if too many detected
    if len(circles[0]) > max_stars:
        for param2_adjust in range(1, 6):
            circles = cv2.HoughCircles(
                image,
                method=cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=250,
                param2=param2_adjust,
                minRadius=2,
                maxRadius=6
            )
            if circles is not None and len(circles[0]) <= max_stars:
                break

    if circles is None:
        return []

    detected_stars = []
    for (x, y, r) in circles[0, :]:
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            brightness = int(image[int(y), int(x)])  # (row, col) = (y, x)
            detected_stars.append((int(x), int(y), int(r), brightness))  # NO +4 here!

    return detected_stars

def save_star_data(filepath: str, stars: list):
    """
    Saves detected stars' coordinates and properties into a text file, adding a star ID.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        file.write("ID , x , y , r , b\n")
        for idx, star in enumerate(stars, start=1):
            file.write(f"{idx} , {star[0]} , {star[1]} , {star[2]} , {star[3]}\n")

def load_and_prepare_image(path: str, target_size: tuple) -> np.ndarray:
    """
    Loads and resizes an image.

    Args:
        path (str): Path to the image.
        target_size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Grayscale image array.
    """
    img = Image.open(path).resize(target_size)
    img_np = np.array(img)
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

def draw_detected_stars(image: np.ndarray, stars: list) -> np.ndarray:
    """
    Draws circles on detected stars (bigger radius only for visualization).

    Args:
        image (np.ndarray): Original BGR image.
        stars (list): List of (x, y, r, b) tuples.

    Returns:
        np.ndarray: Image with stars highlighted.
    """
    for (x, y, r, _) in stars:
        draw_radius = max(r + 4, 6)  # Add +4 only when drawing, not saving
        image = cv2.circle(image, (x, y), draw_radius, (255, 0, 255), thickness=1)
    return image

if __name__ == "__main__":
    # Parameters
    image1_path = './imgs/fr1.jpg'
    image2_path = './imgs/ST_db2.png'
    output_folder = './star_detection_results'
    output1_path = os.path.join(output_folder, 'stars_fr1.txt')
    output2_path = os.path.join(output_folder, 'stars_ST_db2.txt')
    size = (600, 600)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load and prepare images
    img1_gray = load_and_prepare_image(image1_path, size)
    img2_gray = load_and_prepare_image(image2_path, size)

    # Detect stars
    stars_img1 = detect_stars(img1_gray, size)
    stars_img2 = detect_stars(img2_gray, size)

    # Save detected stars
    save_star_data(output1_path, stars_img1)
    save_star_data(output2_path, stars_img2)

    # Draw stars
    img1 = np.array(Image.open(image1_path).resize(size))
    img2 = np.array(Image.open(image2_path).resize(size))
    img1_stars = draw_detected_stars(img1, stars_img1)
    img2_stars = draw_detected_stars(img2, stars_img2)

    # Display results
    cv2.imshow('Detected Stars - Image 1', img1_stars)
    cv2.imshow('Detected Stars - Image 2', img2_stars)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary
    print(f"Image 1 ({image1_path}): Detected {len(stars_img1)} stars.")
    print(f"Image 2 ({image2_path}): Detected {len(stars_img2)} stars.")