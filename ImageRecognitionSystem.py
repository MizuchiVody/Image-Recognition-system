from ColorDetection import ColorLabeler
from ShapeDetection import ShapeDetector
import numpy as np
import imutils
import cv2

# 'C:/Users/user/Desktop/The new ultimate/Computer science/PYTHON programming/ROV Software/shapes_and_colors.jpg'
# 'C:/Users/user/Desktop/The new ultimate/Robotics and mechatronics/UnderWater '
#    'ROV/ROV2018MizuchyVody/ROVsoftware/TailSection2.jpg'
# 'C:/Users/user/Desktop/The new ultimate/Robotics and mechatronics/UnderWater '
#   'ROV/ROV2018MizuchyVody/ROVsoftware/d3d5792c-6a01-4f6e-a329-c029695e5fd2.jpg'
image = cv2.imread(
    'C:/Users/user/Desktop/The new ultimate/Robotics and mechatronics/UnderWater '
    'ROV/ROV2018MizuchyVody/ROVsoftware/TailSection5.jpg', -1)

resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# blur the resized image slightly, then convert it to both
# grayscale and the L*a*b* color spaces
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
HSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV_FULL)

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([190, 255, 255])
HSVblueThresh = cv2.inRange(HSV, lower_blue, upper_blue)

lower_yellow = np.array([25, 50, 50])
upper_yellow = np.array([45, 255, 255])
HSVyellowThresh = cv2.inRange(HSV, lower_yellow, upper_yellow)

lower_red = np.array([3, 50, 50])
upper_red = np.array([9, 255, 200])
HSVredThresh = cv2.inRange(HSV, lower_red, upper_red)

HSVredThresh = cv2.bitwise_and(HSVredThresh, HSVredThresh, mask=HSVredThresh)
HSVblueThresh = cv2.bitwise_and(HSVblueThresh, HSVblueThresh, mask=HSVblueThresh)
HSVyellowThresh = cv2.bitwise_and(HSVyellowThresh, HSVyellowThresh, mask=HSVyellowThresh)
# THRESH_BINARY
# Tweak the args
# Understand the code more

cv2.imshow("Image", HSV)
cv2.waitKey(0)
cv2.imshow("Image", HSVyellowThresh)
cv2.waitKey(0)
cnts = cv2.findContours(HSVyellowThresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# initialize the shape detector and color labeler in initial shape and initial color
iShape = ShapeDetector()
iColor = ColorLabeler()
# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
    else:
        cX = 0
        cY = 0
    print(c)
    # detect the shape of the contour and label the color
    shape = iShape.DetectTheShape(c)
    color = iColor.LabelTheColor(HSV, c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape and labeled
    # color on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    text = "{} {}".format(color, shape)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, text, (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print('HNA')
    # show the output image
cv2.imshow("Image", image)
print('HNA2')
cv2.waitKey(0)
print('HNA3')
