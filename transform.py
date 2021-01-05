import numpy as np
import cv2


def order_points(points):
    rectangle = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rectangle[0] = points[np.argmin(s)]
    rectangle[2] = points[np.argmax(s)]

    difference = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]

    return rectangle


def four_point_transform(image, points):
    rectangle = order_points(points)
    (tl, tr, br, bl) = rectangle

    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(rectangle, destination)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))

    return warped

