import tensorflow as tf

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan


def pix2pix_generator(inputs, weight_decay=2.5e-5, is_training=True):
    noise, vae_generated_images = inputs
    with tf.contrib.framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope(
                [layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(noise, 1024)
            net = layers.fully_connected(net, 7 * 7 * 128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = layers.conv2d(
                net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)
            return net

def _discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
  """Core MNIST discriminator.
  This function is reused between the different GAN modes (unconditional,
  conditional, etc).
  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Labels to optionally condition the network on.
    weight_decay: The L2 weight decay.
  Returns:
    Final fully connected discriminator layer. [batch_size, 1024].
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=_leaky_relu, normalizer_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay),
      biases_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.conv2d(img, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.flatten(net)
    if is_conditional:
      net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
    net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

    return net
