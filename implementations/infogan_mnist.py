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
from datasets.tfrecord_reader import mnist as mnist_reader
from visualization import visual_gan
from models.gan import info_gan as gan_network
import argparse

batch_size = 128
keys = ['rotation', 'width']
image_size = [28, 28, 1]

def load_batch(dataset_path, dataset_name, split_name, batch_size=128, image_size=[64, 64, 3]):
	#1. Data pipeline
	dataset = mnist_reader.get_split(split_name, dataset_path)
	print(dataset_name)
	print(split_name)
	data_provider = slim.dataset_data_provider.DatasetDataProvider(
		            dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
	[image, label] = data_provider.get(['image', 'label'])

	image = (tf.to_float(image) - 128.0) / 128.0 # convert 0~255 scale into -1~1 scale
	image.set_shape(image_size)
	images, labels = tf.train.batch(
		      [image, label],
		      batch_size=batch_size,
		      num_threads=4,
		      capacity=2 * batch_size)
	return dataset, images, labels

def train(checkpoint_path, dataset_path, batch_size, result_path):
	with tf.Graph().as_default():
		
		#1. Data pipeline
		dataset, images, labels = load_batch(dataset_path, 'mnist', 'train', batch_size=batch_size, image_size=image_size)
		one_hot_labels = tf.one_hot(labels, dataset.num_classes)
		print(images.shape)
		# Sanity check that we're getting images.
		imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
		visual_gan.visualize_digits(imgs_to_visualize)


		#2. model deploy

		# Dimensions of the structured and unstructured noise dimensions.
		cat_dim, cont_dim, noise_dims = 10, 2, 64
		unstructured_noise_dims = noise_dims - cont_dim

		generator_fn = functools.partial(gan_network.generator, categorical_dim=cat_dim)
		discriminator_fn = functools.partial(
		    gan_network.discriminator, categorical_dim=cat_dim,
		    continuous_dim=cont_dim)
		unstructured_inputs, structured_inputs = gan_network.get_infogan_noise(
		    batch_size, cat_dim, cont_dim, noise_dims)

		infogan_model = tfgan.infogan_model(
		    generator_fn=generator_fn,
		    discriminator_fn=discriminator_fn,
		    real_data=images,
		    unstructured_generator_inputs=unstructured_inputs,
		    structured_generator_inputs=structured_inputs)


		#3. training op
		infogan_loss = tfgan.gan_loss(
		    infogan_model,
		    gradient_penalty_weight=1.0,
		    mutual_information_penalty_weight=1.0)

		# Sanity check that we can evaluate our losses.
		visual_gan.evaluate_tfgan_loss(infogan_loss)
		generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
		gan_train_ops = tfgan.gan_train_ops(
		    infogan_model,
		    infogan_loss,
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
				for i in range(100000):
					cur_loss, _ = train_step_fn(
					sess, gan_train_ops, global_step, train_step_kwargs={})
					loss_values.append((i, cur_loss))

					if i % 500 == 0:
						visual_gan.varying_categorical_noise(sess, infogan_model, 10, unstructured_noise_dims, cont_dim, i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, infogan_model, 10, 0, unstructured_noise_dims, cont_dim
	    																,i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, infogan_model, 10, 1, unstructured_noise_dims, cont_dim
	    																,i, result_path)

					if i % 1000 == 0: 
						print('Current loss: %f' % cur_loss)
						if not tf.gfile.Exists(checkpoint_path):
							tf.gfile.MakeDirs(checkpoint_path)
						save_dir = os.path.join(checkpoint_path, "infogan"+'_'+str(i)+'.ckpt')
						saver.save(sess, save_dir)
						print("Model saved in file: %s" % checkpoint_path)