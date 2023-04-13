import cv2
import numpy as np

# Load the image
img = cv2.imread('image.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Hough Circle Transform to detect circles
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Check if any circles were found
if circles is not None:
    circles = np.uint16(np.around(circles))

    # Find the two circles
    circles = sorted(circles[0,:], key=lambda circle: -circle[2])
    if len(circles) < 2:
        print("At least two circles were not found.")
        exit()

    # Find the touching point between the circles
    circle1, circle2 = circles[:2]
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    if d > r1 + r2:
        print("The circles are not touching.")
        exit()
    elif d == r1 + r2:
        touching_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    else:
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        touching_point = (int(x1 + a * (x2 - x1) / d), int(y1 + a * (y2 - y1) / d))
        intersection1 = (int(touching_point[0] + h * (y2 - y1) / d), int(touching_point[1] - h * (x2 - x1) / d))
        intersection2 = (int(touching_point[0] - h * (y2 - y1) / d), int(touching_point[1] + h * (x2 - x1) / d))

    # Create a mask
    mask = np.zeros_like(img)
    cv2.circle(mask, (x1, y1), r1, (255, 255, 255), -1)
    cv2.circle(mask, (x2, y2), r2, (255, 255, 255), -1)

    # Remove the smaller circle
    if d == r1 + r2:
        cv2.circle(mask, touching_point, r2, (0, 0, 0), -1)
    else:
        if r1 < r2:
            cv2.circle(mask, touching_point, r1, (0, 0, 0), -1)
            cv2.circle(mask, intersection1, r2, (0, 0, 0), -1)
            cv2.circle(mask,
