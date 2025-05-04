import cv2
import numpy as np

def show_data(src_pts, dst_pts, line1_pts, line1, line2_pts, line2, matches, img1, img2):
    """
    Visualize matched stars and lines on both images.
    """
    img1 = img1.copy()
    img2 = img2.copy()

    # Draw points on lines
    for x, y, *_ in line1_pts:
        cv2.circle(img1, (int(x), int(y)), 3, (0, 255, 0), -1)

    for x, y, *_ in line2_pts:
        cv2.circle(img2, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw match lines between stars
    combined = np.hstack((img1, img2))
    h, w = img1.shape[:2]

    for (src, dst) in matches:
        x1, y1 = src
        x2, y2 = dst
        cv2.line(combined, (int(x1), int(y1)), (int(x2 + w), int(y2)), (255, 0, 255), 1)

    # Draw source sample points
    for x, y, *_ in src_pts:
        cv2.circle(combined, (int(x), int(y)), 5, (255, 255, 0), 2)

    for x, y, *_ in dst_pts:
        cv2.circle(combined, (int(x + w), int(y)), 5, (255, 255, 0), 2)

    cv2.imshow("Matched Stars (Left=Image1, Right=Image2)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
