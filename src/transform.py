## Code based on Pyimaagesearch tutorial
## Source: https://www.pyimagesearch.com/


import numpy as np 
import cv2

def order_points(pts):
	"""
	input(pts): list of four points specifying the (x,y) coordinates of rectangle.

	return(rect): points organized in four points
	"""
	rect = np.zeros((4,2), dtype = "float32") # for x and y value 

	s = pts.sum(axis = 1)
	# min sum of the x,y coordinate
	rect[0] = pts[np.argmin(s)] # top left
	# max sum of the x,y coordinate
	rect[2] = pts[np.argmax(s)] # bottom right

	d = np.diff(pts, axis = 1)
	# min diff of the x,y coordinate
	rect[1] = pts[np.argmin(d)] # top right
	# max diff of the x,y coordinate
	rect[3] = pts[np.argmax(d)] # bottom left

	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# find the width
	widthA = np.sqrt(((br[0] - bl[0]) ** 2)+((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2)+((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# find the height
	heightA = np.sqrt(((tr[0] - br[0]) ** 2)+((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2)+((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA),int(heightB))

	dst = np.array([[0, 0], 
					[maxWidth-1,0], 
					[maxWidth-1, maxHeight-1], 
					[0, maxHeight-1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped