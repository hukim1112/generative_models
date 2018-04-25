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
import cv2
# Main TFGAN library.
tfgan = tf.contrib.gan

# Shortcuts for later.
slim = tf.contrib.slim
layers = tf.contrib.layers
ds = tf.contrib.distributions
from datasets.data_downloader import mnist
from datasets.tfrecord_reader import tfrecord_reader
from visualization import visual_gan
from models.gan import info_gan_64x64 as gan_networks

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path')
# parser.add_argument('--dataset_path')
# args = parser.parse_args()
# batch_size = 128
# checkpoint_path = args.checkpoint_path
# dataset_path = args.dataset_path



def load_batch(dataset_path, dataset_name, split_name, batch_size=128, image_size=[64, 64, 3]):
	#1. Data pipeline
	dataset = tfrecord_reader.get_split(dataset_name, split_name, dataset_path)
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
		
		dataset, images, labels = load_batch(dataset_path, 'figures', 'train', batch_size=batch_size, image_size=[64, 64, 3])
		one_hot_labels = tf.one_hot(labels, dataset.num_classes)
		
		# Sanity check that we're getting images.
		imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
		visual_gan.visualize_digits(imgs_to_visualize)



		#2. model deploy

		# Dimensions of the structured and unstructured noise dimensions.
		cat_dim, cont_dim, noise_dims = 3, 2, 64
		unstructured_noise_dims = noise_dims - cont_dim

		generator_fn = functools.partial(gan_networks.generator, categorical_dim=cat_dim)
		discriminator_fn = functools.partial(
		    gan_networks.discriminator, categorical_dim=cat_dim,
		    continuous_dim=cont_dim)
		unstructured_inputs, structured_inputs = gan_networks.get_infogan_noise(
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
				for i in range(50001):
					cur_loss, _ = train_step_fn(
					sess, gan_train_ops, global_step, train_step_kwargs={})
					loss_values.append((i, cur_loss))
					if i % 500 == 0:
						print('sssssssssssssssssssssssssssssssssssssssssssssssssssss')
						visual_gan.varying_categorical_noise(sess, infogan_model, 3, unstructured_noise_dims, cont_dim, i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, infogan_model, 3, 0, unstructured_noise_dims, cont_dim
    																,i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, infogan_model, 3, 1, unstructured_noise_dims, cont_dim
    																,i, result_path)
					if i % 1000 == 0: 
						print('Current loss: %f' % cur_loss)

						if not tf.gfile.Exists(checkpoint_path):
							tf.gfile.MakeDirs(checkpoint_path)
						save_dir = os.path.join(checkpoint_path, "infogan"+'_'+str(i)+'.ckpt')
						saver.save(sess, save_dir)
						print("Model saved in file: %s" % checkpoint_path)

def impl(checkpoint_path, dataset_path, batch_size):
	with tf.Graph().as_default():

		dataset, images, labels = load_batch(dataset_path, 'figures', 'train', batch_size=batch_size, image_size=[256, 256, 3])
		one_hot_labels = tf.one_hot(labels, dataset.num_classes)

		#2. input define
		# Generate three sets of images to visualize the effect of each of the structured noise
		# variables on the output.

		cat_dim, cont_dim, noise_dims = 3, 2, 64

		rows = 2
		categorical_sample_points = np.arange(0, 3)
		continuous_sample_points = np.linspace(-1.0, 1.0, 10)
		noise_args = (rows, categorical_sample_points, continuous_sample_points,
		              noise_dims-cont_dim, cont_dim)

		display_noises = []
		display_noises.append(gan_networks.get_eval_noise_categorical(*noise_args))
		display_noises.append(gan_networks.get_eval_noise_continuous_dim1(*noise_args))
		display_noises.append(gan_networks.get_eval_noise_continuous_dim2(*noise_args))

		#3. infogan model deploy
		generator_fn = functools.partial(gan_networks.generator, categorical_dim=cat_dim)
		discriminator_fn = functools.partial(
		    gan_networks.discriminator, categorical_dim=cat_dim,
		    continuous_dim=cont_dim)
		unstructured_inputs, structured_inputs = gan_networks.get_infogan_noise(
		    batch_size, cat_dim, cont_dim, noise_dims)

		infogan_model = tfgan.infogan_model(
		    generator_fn=generator_fn,
		    discriminator_fn=discriminator_fn,
		    real_data=images,
		    unstructured_generator_inputs=unstructured_inputs,
		    structured_generator_inputs=structured_inputs)

		#4. display generated images
		display_images = []
		for noise in display_noises:
		    with tf.variable_scope(infogan_model.generator_scope, reuse=True):
		        display_images.append(infogan_model.generator_fn(noise))

		display_img = tfgan.eval.image_reshaper(
		    tf.concat(display_images, 0), num_cols=10)

		saver = tf.train.Saver()

		with tf.Session() as sess:
			with slim.queues.QueueRunners(sess):
				saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
				plt.imshow(np.squeeze(sess.run(display_img*128+128)).astype(np.uint8))
				plt.show()