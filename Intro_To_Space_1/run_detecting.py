import os
from star_detector import detect_stars, save_star_data, load_and_prepare_image, draw_detected_stars
import numpy as np
import cv2
from PIL import Image

# Parameters
image_infos = [
    ('fr1.jpg', 'stars_fr1.txt', 'detected_fr1.png'),
    ('fr2.jpg', 'stars_fr2.txt', 'detected_fr2.png'),
    ('ST_db1.png', 'stars_ST_db1.txt', 'detected_ST_db1.png'),
    ('ST_db2.png', 'stars_ST_db2.txt', 'detected_ST_db2.png')
]
input_folder = './imgs'
output_folder = './star_detection_results'
detected_images_folder = './star_detection_results/visual_results'
size = (600, 600)

# Create output folders if missing
os.makedirs(output_folder, exist_ok=True)
os.makedirs(detected_images_folder, exist_ok=True)

for img_name, txt_name, save_img_name in image_infos:
    input_path = os.path.join(input_folder, img_name)
    output_txt_path = os.path.join(output_folder, txt_name)
    output_img_path = os.path.join(detected_images_folder, save_img_name)

    # Load and prepare grayscale image
    img_gray = load_and_prepare_image(input_path, size)

    # Detect stars
    stars = detect_stars(img_gray, size)

    # Save detected stars to txt
    save_star_data(output_txt_path, stars)

    # Load color image for visualization
    img_color = np.array(Image.open(input_path).resize(size))

    # Draw detected stars on the color image
    img_with_stars = draw_detected_stars(img_color, stars)

    # Save the image with detected stars
    cv2.imwrite(output_img_path, cv2.cvtColor(img_with_stars, cv2.COLOR_RGB2BGR))

    # Optionally show the result
    cv2.imshow(f'Detected Stars - {img_name}', img_with_stars)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print report
    print(f"[{img_name}] Detected {len(stars)} stars. Results saved to {output_txt_path}")
    print(f"Image with detected stars saved to {output_img_path}")
