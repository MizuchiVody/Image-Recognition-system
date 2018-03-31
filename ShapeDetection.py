import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def DetectTheShape(self, Contour):
        # Initialize the shape name and approximate the contour
        shape = ""
        peri = cv2.arcLength(Contour, True)
        approx = cv2.approxPolyDP(Contour, 0.01 * peri, True)  # this statement removes unnecessary points,
        # reducing the curvatures makes it easy to calculate the number of line segments in each shape.
        # The algorithm used is the Ramer-Douglas-Peucker algorithm, aka split-and-merge algorithm
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            print(x, y, w, h)
            ar = w / float(h)

            rect = x / float(y)


            if 0.95 <= ar <= 1.05 and rect >= 0.95 and rect <= 1.05:
                shape = "Rectangle"
            else:
                shape = "something"
        return shape
