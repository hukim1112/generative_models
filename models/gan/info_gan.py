import tensorflow as tf
slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions
import numpy as np
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def generator(inputs, categorical_dim, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Args:
        inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
            noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
            2D, and `inputs[1]` must be 1D. All must have the same first dimension.
        categorical_dim: Dimensions of the incompressible categorical noise.
        weight_decay: The value of the l2 weight decay.
    
    Returns:
        A generated image in the range [-1, 1].
    """
    unstructured_noise, cat_noise, cont_noise = inputs
    cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
    all_noise = tf.concat([unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
    
    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(all_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)
    
        return net

def discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        categorical_dim: Dimensions of the incompressible categorical noise.
        continuous_dim: Dimensions of the incompressible continuous noise.
    
    Returns:
        Logits for the probability that the image is real, and a list of posterior
        distributions for each of the noise vectors.
    """
    with slim.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    
        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        # Recognition network for latent variables has an additional layer
        encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        logits_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        q_cat = ds.Categorical(logits_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        mu_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        sigma_cont = tf.ones_like(mu_cont)
        q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]


def get_infogan_noise(batch_size, categorical_dim, structured_continuous_dim,
                      total_continuous_noise_dims):
  """Get unstructured and structured noise for InfoGAN.

  Args:
    batch_size: The number of noise vectors to generate.
    categorical_dim: The number of categories in the categorical noise.
    structured_continuous_dim: The number of dimensions of the uniform
      continuous noise.
    total_continuous_noise_dims: The number of continuous noise dimensions. This
      number includes the structured and unstructured noise.

  Returns:
    A 2-tuple of structured and unstructured noise. First element is the
    unstructured noise, and the second is a 2-tuple of
    (categorical structured noise, continuous structured noise).
  """
  # Get unstructurd noise.
  unstructured_noise = tf.random_normal(
      [batch_size, total_continuous_noise_dims - structured_continuous_dim])

  # Get categorical noise Tensor.
  categorical_dist = ds.Categorical(logits=tf.zeros([categorical_dim]))
  categorical_noise = categorical_dist.sample([batch_size])

  # Get continuous noise Tensor.
  continuous_dist = ds.Uniform(-tf.ones([structured_continuous_dim]),
                               tf.ones([structured_continuous_dim]))
  continuous_noise = continuous_dist.sample([batch_size])

  return [unstructured_noise], [categorical_noise, continuous_noise]

# (joelshor): Refactor the `eval_noise` functions to reuse code.
def get_eval_noise_categorical(
    noise_samples, categorical_sample_points, continuous_sample_points,
    unstructured_noise_dims, continuous_noise_dims):
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
  rows, cols = noise_samples, len(categorical_sample_points)

  # Take random draws for non-categorical noise, making sure they are constant
  # across columns.
  unstructured_noise = []
  for _ in range(rows):
    cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
    unstructured_noise.extend([cur_sample] * cols)
  unstructured_noise = np.concatenate(unstructured_noise)

  continuous_noise = []
  for _ in range(rows):
    cur_sample = np.random.choice(
        continuous_sample_points, size=[1, continuous_noise_dims])
    continuous_noise.extend([cur_sample] * cols)
  continuous_noise = np.concatenate(continuous_noise)

  # Increase categorical noise from left to right, making sure they are constant
  # across rows.
  categorical_noise = np.tile(categorical_sample_points, rows)

  return unstructured_noise, categorical_noise, continuous_noise


def get_eval_noise_continuous_dim1(
    noise_samples, categorical_sample_points, continuous_sample_points,
    unstructured_noise_dims, continuous_noise_dims):  # pylint:disable=unused-argument
  """Create noise showing impact of first dim continuous noise in InfoGAN.

  First dimension of continuous noise is constant across columns. Other noise is
  constant across rows.

  Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

  Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays.
  """
  rows, cols = noise_samples, len(continuous_sample_points)

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

  cont_noise_dim2 = []
  for _ in range(rows):
    cur_sample = np.random.choice(continuous_sample_points, size=[1, 1])
    cont_noise_dim2.extend([cur_sample] * cols)
  cont_noise_dim2 = np.concatenate(cont_noise_dim2)

  # Increase first dimension of continuous noise from left to right, making sure
  # they are constant across rows.
  cont_noise_dim1 = np.expand_dims(np.tile(continuous_sample_points, rows), 1)

  continuous_noise = np.concatenate((cont_noise_dim1, cont_noise_dim2), 1)

  return unstructured_noise, categorical_noise, continuous_noise


def get_eval_noise_continuous_dim2(
    noise_samples, categorical_sample_points, continuous_sample_points,
    unstructured_noise_dims, continuous_noise_dims):  # pylint:disable=unused-argument
  """Create noise showing impact of second dim of continuous noise in InfoGAN.

  Second dimension of continuous noise is constant across columns. Other noise
  is constant across rows.

  Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

  Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays.
  """
  rows, cols = noise_samples, len(continuous_sample_points)

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

  cont_noise_dim1 = []
  for _ in range(rows):
    cur_sample = np.random.choice(continuous_sample_points, size=[1, 1])
    cont_noise_dim1.extend([cur_sample] * cols)
  cont_noise_dim1 = np.concatenate(cont_noise_dim1)

  # Increase first dimension of continuous noise from left to right, making sure
  # they are constant across rows.
  cont_noise_dim2 = np.expand_dims(np.tile(continuous_sample_points, rows), 1)

  continuous_noise = np.concatenate((cont_noise_dim1, cont_noise_dim2), 1)

  return unstructured_noise, categorical_noise, continuous_noise