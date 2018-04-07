
import numpy as np
from image_processing import draw, transform
import os
import cv2
#random function
#use np.random.rand(1) to get samples from uniform distribution over [0, 1]


def draw_canvas(image_size=(256, 256, 3), color=None):
	canvas =  np.zeros(image_size, dtype = np.uint8)
	if color != None:
		canvas[:, :, 0] = color[0]
		canvas[:, :, 1] = color[1]
		canvas[:, :, 2] = color[2]
	return canvas

def draw_rectangles(dataset_dir, image_size=(256, 256, 3), color = (255, 0, 0), background = (0, 0, 0), num=10000):
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values * 60
		width = int(image_size[0]/8)
		height = width * 3
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)
		print(canvas.shape)	
		#Draw the rectangle into canvas
		draw.rectangle(canvas, pt1, pt2, color=color, thickness = -1)
		#transform the image
		img = transform.rotate(canvas, center, angle, 1, border_color=background)
		cv2.imwrite(img, os.path.join(dataset_dir, 'rect_'+str(i)+'.png'))

def draw_eclipses(dataset_dir, image_size=(256, 256, 3), color = (0, 255, 0), background = (0, 0, 0), num=10000):
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values * 60
		width = int(image_size[0]/8)
		height = width * 3
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)
		#Draw the rectangle into canvas
		draw.eclipse(canvas, center, width, height, angle = angle, color=color, thickness = -1)
		#transform the image
		
		cv2.imwrite(img, os.path.join(dataset_dir, 'eclipse_'+str(i)+'.png'))

def draw_triangles(dataset_dir, image_size=(256, 256, 3), color = (0, 0, 255), background = (0, 0, 0), num=10000):
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values * 60
		width = int(image_size[0]/8)
		height = width * 3
		pt1 = (center[0], int(center[1]-height/2))
		pt2 = (int(center[0]-width/2), int(center[1] + height/2) )
		pt3 = (int(center[0]+width/2), int(center[1] + height/2) )
		pt_list = [pt1, pt2, pt3]
		#Draw the rectangle into canvas
		draw.polygon(canvas, pt_list, color=color, thickness = -1)
		#transform the image
		img = transform.rotate(canvas, center, angle, 1, border_color=background)
		cv2.imwrite(img, os.path.join(dataset_dir, 'triangle'+str(i)+'.png'))	



def run(dataset_dir):
	os.makedirs(dataset_dir, exist_ok = True)

	draw_rectangles(dataset_dir, num=10)
	draw_eclipses(dataset_dir, num=10)
	draw_triangles(dataset_dir, num=10)