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
from models.gan import conditional_gan


tf.reset_default_graph()
batch_size = 32
dataset = tfrecord_reader.get_split('mnist', 'train', '/home/dan/prj/datasets')
data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=2*batch_size, common_queue_min=batch_size)    
[image, label] = data_provider.get(['image', 'label'])

image = (tf.to_float(image) - 128.0) / 128.0
images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=4,
      capacity=5 * batch_size)
one_hot_labels = tf.one_hot(labels, dataset.num_classes)

noise_dims = 64
conditional_gan_model = tfgan.gan_model(
    generator_fn=conditional_gan.conditional_generator_fn,
    discriminator_fn=conditional_gan.conditional_discriminator_fn,
    real_data=images,
    generator_inputs=(tf.random_normal([batch_size, noise_dims]), 
                      one_hot_labels))

gan_loss = tfgan.gan_loss(
    conditional_gan_model, gradient_penalty_weight=1.0)

# Sanity check that we can evaluate our losses.
visual_gan.evaluate_tfgan_loss(gan_loss)

generator_optimizer = tf.train.AdamOptimizer(0.0009, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    conditional_gan_model,
    gan_loss,
    generator_optimizer,
    discriminator_optimizer)

# Set up class-conditional visualization. We feed class labels to the generator
# so that the the first column is `0`, the second column is `1`, etc.
images_to_eval = 500
assert images_to_eval % 10 == 0

random_noise = tf.random_normal([images_to_eval, 64])
one_hot_labels = tf.one_hot(
    [i for _ in range(images_to_eval // 10) for i in range(10)], depth=10) 
with tf.variable_scope(conditional_gan_model.generator_scope, reuse=True):
    eval_images = conditional_gan_model.generator_fn(
        (random_noise, one_hot_labels))
reshaped_eval_imgs = tfgan.eval.image_reshaper(
    eval_images[:20, ...], num_cols=10)



global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps()
loss_values, xent_score_values  = [], []



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        start_time = time.time()
        for i in range(5001):
            if i % 10 == 0:
                print(i)
            cur_loss, _ = train_step_fn(
                sess, gan_train_ops, global_step, train_step_kwargs={})
            loss_values.append((i, cur_loss))
            if i % 1000 == 0:
                print('Current loss: %f' % cur_loss)
                #visual_gan.visualize_training_generator(i, start_time, sess.run(reshaped_eval_imgs))
                plt.imsave(os.path.join('results','cgan_'+str(i)+'.png')
                    ,np.squeeze(sess.run(reshaped_eval_imgs)), cmap='gray')