import cv2

def crop(img, center, width, height):
	return cv2.getRectSubPix(img, (width, height), center)

def translate(img, x, y, border_size=None, border_color=None):
	if border_size == None:
		border_size = (img.shape[1], img.shape[0])
	if border_color == None:
		border_color=(255, 255, 255)
	return cv2.warpAffine(img, np.array([[1, 0, x], [0, 1, y]]), border_size, borderValue=border_color, dtype = np.float)

def rotate(img, center, angle, scale=1.0, border_size=None, border_color=None):
	'''The angle is degree scale(like 30, 45, 90). if angle = 0, you can use this as resize funtion with rotation available'''
	if border_size == None:
		border_size = (img.shape[1], img.shape[0])
	if border_color == None:
		border_color=(255, 255, 255)
	M = cv2.getRotationMatrix2D(center, angle, scale)
	return cv2.warpAffine(img, M, border_size, borderValue=border_color)


def bordered_resize(img, scale, center=None, border_size=None, border_color=None):
	if center == None:
		center = (int(img.shape[1]), int(img.shape[0]))
	if border_size == None:
		border_size = (img.shape[1], img.shape[0])
	if border_color == None:
		border_color=(255, 255, 255)
	M = cv2.getRotationMatrix2D(center, 0, scale)
	return cv2.warpAffine(img, M, border_size, borderValue=border_color)

def resize(img, width, height):
	return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
