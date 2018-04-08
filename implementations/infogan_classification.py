# nfoGAN is a technique to induce semantic meaning in the latent space of a GAN generator in an unsupervised way. In this example, the generator learns how to generate a specific digit without ever seeing labels. 
#This is achieved by maximizing the mutual information between some subset of the noise vector and the generated images, while also trying to generate realistic images. 
#See InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets by Chen at al for more details.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import functools

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
# parser.add_argument('-f', '--my-foo', default='foobar')
# parser.add_argument('-b', '--bar-value', default=3.14)
parser.add_argument('--checkpoint_path')
parser.add_argument('--dataset_path')
args = parser.parse_args()
batch_size = 32
checkpoint_path = args.checkpoint_path
dataset_path = args.dataset_path



with tf.Graph().as_default():
	#1. Data pipeline
	dataset = tfrecord_reader.get_split('mnist', 'train', dataset_path)
	data_provider = slim.dataset_data_provider.DatasetDataProvider(
	            dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
	[image, label] = data_provider.get(['image', 'label'])

	image = (tf.to_float(image) - 128.0) / 128.0
	images, labels = tf.train.batch(
	      [image, label],
	      batch_size=batch_size,
	      num_threads=4,
	      capacity=2 * batch_size)
	one_hot_labels = tf.one_hot(labels, dataset.num_classes)

	#2. input define
	# Generate three sets of images to visualize the effect of each of the structured noise
	# variables on the output.

	cat_dim, cont_dim, noise_dims = 10, 2, 64

	rows = 2
	categorical_sample_points = np.arange(0, 10)
	continuous_sample_points = np.linspace(-1.0, 1.0, 10)
	noise_args = (rows, categorical_sample_points, continuous_sample_points,
	              noise_dims-cont_dim, cont_dim)

	display_noises = []
	display_noises.append(info_gan.get_eval_noise_categorical(*noise_args))
	display_noises.append(info_gan.get_eval_noise_continuous_dim1(*noise_args))
	display_noises.append(info_gan.get_eval_noise_continuous_dim2(*noise_args))

	#3. infogan model deploy
	generator_fn = functools.partial(info_gan.generator, categorical_dim=cat_dim)
	discriminator_fn = functools.partial(
	    info_gan.discriminator, categorical_dim=cat_dim,
	    continuous_dim=cont_dim)
	unstructured_inputs, structured_inputs = info_gan.get_infogan_noise(
	    batch_size, cat_dim, cont_dim, noise_dims)

	infogan_model = info_gan.infogan_model(
	    generator_fn=generator_fn,
	    discriminator_fn=discriminator_fn,
	    real_data=images,
	    unstructured_generator_inputs=unstructured_inputs,
	    structured_generator_inputs=structured_inputs)

	#4. classfication of images
	
	for noise in display_noises:
		print(len(noise))
		with tf.variable_scope(infogan_model.discriminator_scope, reuse=True):
		    logits_real, [q_cat, q_cont] = discriminator_fn(generator_fn(noise), None)




	saver = tf.train.Saver()

	with tf.Session() as sess:
		with slim.queues.QueueRunners(sess):
			saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
			print('cat : ', sess.run(q_cat))
			print('label : ', sess.run(labels))