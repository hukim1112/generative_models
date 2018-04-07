import cv2
import numpy as np


def circle(img, center, radius, color, thickness):
	'''EX)circle(imm, (200, 100), 20, color = (0, 0, 255), thickness=5)
	If you want to fill it with the color, thickness should be -1'''
	cv2.circle(img, center, radius, color, thickness)

def rectangle(img, topleft, bottomright, color, thickness ):
	'''EX )rectangle(imm, (100, 50), (150, 120), color = (0, 255, 0), thickness =1)
	If you want to fill it with the color, thickness should be -1'''
	cv2.rectangle(img, topleft, bottomright, color, thickness)

def polylines(img, pointlist, color, thickness):
	'''EX )polylines(img, [[10, 20], [40, 50], [30, 15]], color = (0, 255, 0), thickness =1)
	If you want to fill it with the color, thickness should be -1'''
	pts = np.array(pointlist, np.int32)
	pts = pts.reshape((-1, 1, 2))
	cv2.polylines(img, [pts], True, color, thickness= thickness)

def fillpoly(img, pointlist, color):
	pts = np.array(pointlist, np.int32)
	pts = pts.reshape((-1, 1, 2))
	cv2.fillPoly(img, [pts], color)

def eclipse(img, center, major_axis, minor_axis, angle, color, thickness, start_angle=0, end_angle=360):
	'''EX )ellipse(img,(256,256),(100,50),0,0,180,255,-1)
	If you want to fill it with the color, thickness should be -1
	reference : https://docs.opencv.org/3.1.0/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69'''
	cv2.ellipse(img, center, (major_axis, minor_axis), angle,start_angle,end_angle,color,thickness)

def line(img, pt1, pt2, color, thickness):
	cv2.line(img, pt1, pt2, color, thickness)

def text(img, string, bottomleft, scale, color, thickness):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, string, bottomleft, font, scale, color, thickness, cv2.LINE_AA)
