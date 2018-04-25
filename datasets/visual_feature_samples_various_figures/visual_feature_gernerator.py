
import numpy as np
from image_processing import draw, transform
import os
import cv2
import math
#random function
#use np.random.rand(1) to get samples from uniform distribution over [0, 1]


def draw_canvas(image_size=(256, 256, 3), color=None):
	canvas =  np.zeros(image_size, dtype = np.uint8)
	if color != None:
		canvas[:, :, 0] = color[0]
		canvas[:, :, 1] = color[1]
		canvas[:, :, 2] = color[2]
	return canvas

def draw_rectangles(dataset_dir, image_size=(128, 128, 3), color = (200, 50, 50), background = (100, 100, 100), num=20):
	path = os.path.join(dataset_dir)
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas1 = draw_canvas(image_size, color = background)
		canvas2 = draw_canvas(image_size, color = background)
		canvas3 = draw_canvas(image_size, color = background)
		canvas4 = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = list(map(lambda x : math.fabs(x), np.random.normal(loc=0, scale = 0.5, size=[2]) ))
		#random_values = (np.random.rand(2) - 0.5)*2
		center0 = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		center1 = (int(  image_size[0]/2 + (image_size[0]/5)  ), int(image_size[1]/2) )
		center2 = (-int(  image_size[0]/2 +(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values[1] * 60
		width = int(image_size[0]/8)
		height = width * 3
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)

		#Draw the rectangle into canvas
		draw.rectangle(canvas1, pt1, pt2, color=color, thickness = -1)
		#transform the image
		img1 = transform.rotate(canvas1, center1, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'transition', 'left', 'rect_'+str(i)+'.png'), img1[:,:,::-1])

		#Draw the rectangle into canvas
		draw.rectangle(canvas2, pt1, pt2, color=color, thickness = -1)
		#transform the image
		img2 = transform.rotate(canvas2, center2, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'transition', 'right', 'rect_'+str(i)+'.png'), img2[:,:,::-1])

		draw.rectangle(canvas3, pt1, pt2, color=color, thickness = -1)
		
		#transform the image
		img3 = transform.rotate(canvas3, center0, -60, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'rotation', 'left', 'rect_'+str(i)+'.png'), img3[:,:,::-1])

		draw.rectangle(canvas4, pt1, pt2, color=color, thickness = -1)
		#transform the image
		img4 = transform.rotate(canvas4, center0, 60, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'rotation', 'right', 'rect_'+str(i)+'.png'), img4[:,:,::-1])


def draw_eclipses(dataset_dir, image_size=(128, 128, 3), color = (50, 200, 50), background = (100, 100, 100), num=20):
	path = os.path.join(dataset_dir)
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas1 = draw_canvas(image_size, color = background)
		canvas2 = draw_canvas(image_size, color = background)
		canvas3 = draw_canvas(image_size, color = background)
		canvas4 = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = list(map(lambda x : math.fabs(x), np.random.normal(loc=0, scale = 0.5, size=[2]) ))
		#random_values = (np.random.rand(2) - 0.5)*2
		center0 = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		center1 = (int(  image_size[0]/2 + (image_size[0]/5)  ), int(image_size[1]/2) )
		center2 = (-int(  image_size[0]/2 + (image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values[1] * 60
		width = int(image_size[0]/5)
		height = width * 2
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)

		#Draw the rectangle into canvas
		draw.eclipse(canvas1, center1, width, height, angle = angle, color=color, thickness = -1)
		#transform the image
		cv2.imwrite(os.path.join(path, 'transition', 'left', 'eclipse_'+str(i)+'.png'), canvas1[:,:,::-1])

		#Draw the rectangle into canvas
		draw.eclipse(canvas2, center2, width, height, angle = angle, color=color, thickness = -1)
		#transform the image
		cv2.imwrite(os.path.join(path, 'transition', 'right', 'eclipse_'+str(i)+'.png'), canvas2[:,:,::-1])

		#Draw the rectangle into canvas
		draw.eclipse(canvas3, center0, width, height, angle = -60, color=color, thickness = -1)
		#transform the image
		cv2.imwrite(os.path.join(path, 'rotation', 'left', 'eclipse_'+str(i)+'.png'), canvas2[:,:,::-1])

		#Draw the rectangle into canvas
		draw.eclipse(canvas3, center0, width, height, angle = 60, color=color, thickness = -1)
		#transform the image
		cv2.imwrite(os.path.join(path, 'rotation', 'right', 'eclipse_'+str(i)+'.png'), canvas2[:,:,::-1])


def draw_triangles(dataset_dir, image_size=(128, 128, 3), color = (50, 50, 200), background = (100, 100, 100), num=20):
	path = os.path.join(dataset_dir)
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas1 = draw_canvas(image_size, color = background)
		canvas2 = draw_canvas(image_size, color = background)
		canvas3 = draw_canvas(image_size, color = background)
		canvas4 = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = list(map(lambda x : math.fabs(x), np.random.normal(loc=0, scale = 0.5, size=[2]) ))
		#random_values = (np.random.rand(2) - 0.5)*2
		center0 = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		center1 = (int(  image_size[0]/2 + (image_size[0]/5)  ), int(image_size[1]/2) )
		center2 = (-int(  image_size[0]/2 + (image_size[0]/5)  ), int(image_size[1]/2) )

		angle = random_values[1] * 60
		width = int(image_size[0]/4)
		height = width * 3
		pt1 = (center[0], int(center[1]-height/2))
		pt2 = (int(center[0]-width/2), int(center[1] + height/2) )
		pt3 = (int(center[0]+width/2), int(center[1] + height/2) )
		pt_list = [pt1, pt2, pt3]
		
		#Draw the rectangle into canvas
		draw.fillpoly(canvas1, pt_list, color=color)
		#transform the image
		img1 = transform.rotate(canvas1, center1, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'transition', 'left', 'triangle'+str(i)+'.png'), img1[:,:,::-1])	

		#Draw the rectangle into canvas
		draw.fillpoly(canvas2, pt_list, color=color)
		#transform the image
		img2 = transform.rotate(canvas2, center2, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'transition', 'right', 'triangle'+str(i)+'.png'), img1[:,:,::-1])	


		#Draw the rectangle into canvas
		draw.fillpoly(canvas3, pt_list, color=color)
		#transform the image
		img3 = transform.rotate(canvas3, center0, -60, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'rotation', 'left', 'triangle'+str(i)+'.png'), img3[:,:,::-1])	

		#Draw the rectangle into canvas
		draw.fillpoly(canvas4, pt_list, color=color)
		#transform the image
		img4 = transform.rotate(canvas4, center0, 60, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'rotation', 'right', 'triangle'+str(i)+'.png'), img4[:,:,::-1])	


def run(dataset_dir):
	draw_rectangles(dataset_dir, image_size=(64, 64, 3))
	draw_eclipses(dataset_dir, image_size=(64, 64, 3))
	draw_triangles(dataset_dir, image_size=(64, 64, 3))