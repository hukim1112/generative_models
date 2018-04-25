from matplotlib import pyplot as plt
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import time
import cv2
import os
tfgan = tf.contrib.gan
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)
  
def varying_categorical_noise(sess, gan_model, category_number,
    unstructured_noise_dims, continuous_noise_dims, iteration, result_path):
    """Create noise showing impact of categorical noise in InfoGAN.

    Categorical noise is constant across columns. Other noise is constant across
    rows.

    Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

    Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays. Each
    should have shape [noise_samples, ?].
    """
    row_num = 10
    categorical_sample_points = np.array(range(category_number))
    continuous_sample_points = np.linspace(-1.0, 1.0, 10)

    rows, cols = row_num, len(categorical_sample_points)

    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    unstructured_noise = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)
      
    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    continuous_noise = []
    for _ in range(rows):
        cur_sample = np.random.choice(
            continuous_sample_points, size=[1, continuous_noise_dims])
        continuous_noise.extend([cur_sample] * cols)
    continuous_noise = np.concatenate(continuous_noise)

    # Increase categorical noise from left to right, making sure they are constant
    # across rows.
    categorical_noise = np.tile(categorical_sample_points, rows)

    display_noises = [(unstructured_noise, categorical_noise, continuous_noise)]

    display_images = []
    for noise in display_noises:
        with tf.variable_scope(gan_model.generator_scope, reuse=True):
            display_images.append(gan_model.generator_fn(noise))

    display_img = tfgan.eval.image_reshaper(tf.concat(display_images, 0), num_cols=cols)
    results = np.squeeze(sess.run(display_img))
    results = results*128 + 128
    cv2.imwrite(os.path.join(result_path , str(iteration)+'_categorization.png'), results.astype(np.uint8))
    print(str(iteration)+'th result saved')



def varying_noise_continuous_ndim(sess, gan_model, category_number, continuous_order, unstructured_noise_dims, continuous_noise_dims
    ,iteration, result_path):
    """Create noise showing impact of first dim continuous noise in InfoGAN.

      First dimension of continuous noise is constant across columns. Other noise is
      constant across rows.

      Args:
        categorical_sample_points: Possible categorical noise points to sample.
        continuous_sample_points: Possible continuous noise points to sample.
        unstructured_noise_dims: Dimensions of the unstructured noise.
        continuous_noise_dims: Dimensions of the continuous noise.

      Returns:
        Unstructured noise, categorical noise, continuous noise numpy arrays.
    """
    row_num = 10
    categorical_sample_points = np.array(range(category_number))
    continuous_sample_points = np.linspace(-1.0, 1.0, 10)

    rows, cols = row_num, len(continuous_sample_points)

    # Take random draws for non-first-dim-continuous noise, making sure they are
    # constant across columns.
    unstructured_noise = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    categorical_noise = []
    for _ in range(rows):
        cur_sample = np.random.choice(categorical_sample_points)
        categorical_noise.extend([cur_sample] * cols)
    categorical_noise = np.array(categorical_noise)
    #print(categorical_noise)
    continuous_noise = []
    for _ in range(rows):
        cur_sample = np.random.choice(continuous_sample_points, size=[1, continuous_noise_dims])
        continuous_noise.extend([cur_sample] * cols)
    continuous_noise = np.concatenate(continuous_noise)

    # Increase first dimension of continuous noise from left to right, making sure
    # they are constant across rows.
    cont_noise_chosen = np.expand_dims(np.tile(continuous_sample_points, rows), 0)
    continuous_noise[:, continuous_order] = cont_noise_chosen
    

    display_noises = [(unstructured_noise, categorical_noise, continuous_noise)]

    display_images = []
    for noise in display_noises:
        with tf.variable_scope(gan_model.generator_scope, reuse=True):
            display_images.append(gan_model.generator_fn(noise))

    display_img = tfgan.eval.image_reshaper(tf.concat(display_images, 0), num_cols=cols)
    results = np.squeeze(sess.run(display_img))
    results = results*128 + 128
    cv2.imwrite(os.path.join(result_path , str(iteration)+'_continuous'+str(continuous_order)+'.png'), results.astype(np.uint8))
    print(str(iteration)+'_continuous'+str(continuous_order)+'.png' + ' result saved')



def visualize_training_generator(train_step_num, start_time, data_np):
    """Visualize generator outputs during training.
    
    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    plt.show()

def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.
    
    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')
    return images_np

def evaluate_tfgan_loss(gan_loss, name=None):
    """Evaluate GAN losses. Used to check that the graph is correct.
    
    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f'% (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f'% dis_loss_np)