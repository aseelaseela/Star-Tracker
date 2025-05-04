import numpy as np
import random

def ransac_line_fit(stars, iterations=500, threshold=5):
    best_line = None
    best_inliers = []
    
    for _ in range(iterations):
        sample = random.sample(stars, 2)
        (x1, y1) = sample[0][:2]
        (x2, y2) = sample[1][:2]
        
        if x1 == x2 and y1 == y2:
            continue

        # line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2

        inliers = []
        for star in stars:
            x0, y0 = star[:2]
            dist = abs(a*x0 + b*y0 + c) / (np.sqrt(a**2 + b**2) + 1e-6)
            if dist < threshold:
                inliers.append(star)

        if len(inliers) > len(best_inliers):
            best_line = (a, b, c)
            best_inliers = inliers

    return best_line, best_inliers