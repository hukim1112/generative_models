import cv2

def rgb2gray(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def rgb2hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
def gray2rgb(img):
	return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)