import cv2
import numpy as np

cap = cv2.VideoCapture(0)

color_ranges = {
    "red": {
        "lower": np.array([0, 0, 180]),
        "upper": np.array([10, 10, 255])
    },
    "green": {
        "lower": np.array([0, 100, 0]),
        "upper": np.array([80, 255, 80])
    },
    "blue": {
        "lower": np.array([100, 80, 0]),
        "upper": np.array([255, 150, 50])
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([50, 255, 255])
    },
    "purple": {
        "lower": np.array([130, 0, 100]),
        "upper": np.array([255, 50, 255])
    },
}

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 13)
    cv2.imshow("Thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            square = w * h
            print("Ширина контура %f, длина контура %f.Площадь контура составляет %f" %(w, h, square))

            if square >= 10000:  
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)

                color = "unknown"
                mean_color = np.mean(img[y:y + h, x:x + w], axis=(0, 1))

                for name, color_range in color_ranges.items():
                    if np.all((mean_color >= color_range["lower"]) & (mean_color <= color_range["upper"])):
                        color = name
                        break

                cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
                cv2.putText(img, color, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Cubes", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()