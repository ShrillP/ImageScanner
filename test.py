from transform import four_point_transform as fpt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

warped = fpt(image, pts)
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

# python test.py --image /Users/shrillpatel/Desktop/ResumeProjects/ImageScanner/1.jpg --coords "[(141,337), (379,351), (402,557), (159,570)]"