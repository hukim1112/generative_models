# InfoGAN is a technique to induce semantic meaning in the latent space of a GAN generator in an unsupervised way. In this example, the generator learns how to generate a specific digit without ever seeing labels. 
#This is achieved by maximizing the mutual information between some subset of the noise vector and the generated images, while also trying to generate realistic images. 
#See InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets by Chen at al for more details.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import os
import tensorflow as tf

# Main TFGAN library.
tfgan = tf.contrib.gan

# Shortcuts for later.
slim = tf.contrib.slim
layers = tf.contrib.layers
ds = tf.contrib.distributions
from datasets.data_downloader import mnist
from datasets.tfrecord_reader import tfrecord_reader
from visualization import visual_gan
from models.gan import info_gan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path')
parser.add_argument('--dataset_path')
args = parser.parse_args()
batch_size = 128
checkpoint_path = args.checkpoint_path
dataset_path = args.dataset_path





with tf.Graph().as_default():
	
	#1. Data pipeline
	dataset = tfrecord_reader.get_split('mnist', 'train', dataset_path)
	data_provider = slim.dataset_data_provider.DatasetDataProvider(
	            dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
	[image, label] = data_provider.get(['image', 'label'])

	image = (tf.to_float(image) - 128.0) / 128.0 # convert 0~255 scale into 0~1 scale
	images, labels = tf.train.batch(
	      [image, label],
	      batch_size=batch_size,
	      num_threads=4,
	      capacity=2 * batch_size)
	one_hot_labels = tf.one_hot(labels, dataset.num_classes)
	
	#Todo : take images for visual feature information
	visual_feature_images = []
	visual_feature_images['rotation'] = []
	visual_feature_images['width'] = []


	# Sanity check that we're getting images.
	imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
	visual_gan.visualize_digits(imgs_to_visualize)



	#2. model deploy

	# Dimensions of the structured and unstructured noise dimensions.
	cat_dim, cont_dim, noise_dims = 10, 2, 64

	generator_fn = functools.partial(info_gan.generator, categorical_dim=cat_dim)
	discriminator_fn = functools.partial(
	    info_gan.discriminator, categorical_dim=cat_dim,
	    continuous_dim=cont_dim)
	unstructured_inputs, structured_inputs = info_gan.get_infogan_noise(
	    batch_size, cat_dim, cont_dim, noise_dims)

	#todo: megan_model need to be designed. It must have the part of visual feature check.
	megan_model = tfgan.megan_model(
	    generator_fn=generator_fn,
	    discriminator_fn=discriminator_fn,
	    real_data=images,
	    unstructured_generator_inputs=unstructured_inputs,
	    structured_generator_inputs=structured_inputs,
	    visual_feature = visual_feature_images)



	#Todo : I need to design loss function for megan
	#3. training op
	megan_loss = tfgan.gan_loss(
	    infogan_model,
	    gradient_penalty_weight=1.0,
	    mutual_information_penalty_weight=1.0,
	    visual_feature_regularizer_weight=1.0)

	# Sanity check that we can evaluate our losses.
	visual_gan.evaluate_tfgan_loss(megan_loss)
	generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
	discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
	gan_train_ops = tfgan.gan_train_ops(
	    infogan_model,
	    megan_loss,
	    generator_optimizer,
	    discriminator_optimizer)

	#4. Session run learning op

	global_step = tf.train.get_or_create_global_step()
	train_step_fn = tfgan.get_sequential_train_steps()
	loss_values, mnist_score_values  = [], []
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		with slim.queues.QueueRunners(sess):
			start_time = time.time()
			for i in range(20000):
				cur_loss, _ = train_step_fn(
				sess, gan_train_ops, global_step, train_step_kwargs={})
				loss_values.append((i, cur_loss))
				if i % 1000 == 0: 
					print('Current loss: %f' % cur_loss)
					if not tf.gfile.Exists(checkpoint_path):
						tf.gfile.MakeDirs(checkpoint_path)
					save_dir = os.path.join(checkpoint_path, "megan"+'_'+str(i)+'.ckpt')
					saver.save(sess, save_dir)
					print("Model saved in file: %s" % checkpoint_path)