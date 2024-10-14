import cv2
import numpy as np

img = cv2.imread("assets/opencv/find_cubes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color_ranges = {
    "red": {
        "lower": np.array([0, 0, 180]),  # Lower bound for red
        "upper": np.array([10, 10, 255]) # Upper bound for red
    },
    "green": {
        "lower": np.array([0, 100, 0]),  # Lower bound for green
        "upper": np.array([80, 255, 80]) # Upper bound for green
    },
    "blue": {
        "lower": np.array([100, 80, 0]),  # Lower bound for blue
        "upper": np.array([255, 150, 50]) # Upper bound for blue
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),  # Lower bound for yellow
        "upper": np.array([50, 255, 255]) # Upper bound for yellow
    },
    "purple": {
        "lower": np.array([130, 0, 100]),  # Lower bound for purple
        "upper": np.array([255, 50, 255]) # Upper bound for purple
    },
}

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        color = "unknown"
        for name, color_range in color_ranges.items():
            mean_color = np.mean(img[y:y + h, x:x + w], axis=(0, 1))
            print(mean_color)
            if np.all((mean_color >= color_range["lower"]) & (mean_color <= color_range["upper"])):
                color = name
                break
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        cv2.putText(img, color, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.imshow("Cubes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()