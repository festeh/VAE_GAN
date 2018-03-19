import tensorflow as tf
from tensorflow.contrib import layers


def generator(inputs,  weight_decay=2.5e-5, is_training=True):
    noise, _ = inputs
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


def discriminator(img, inputs, weight_decay=2.5e-5):
    _, img_rec = inputs
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn= lambda x: tf.nn.leaky_relu(x, alpha=0.01), normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

        add_net = layers.conv2d(img_rec, 32, [4, 4], stride=2)
        add_net = layers.conv2d(add_net, 64, [4, 4], stride=2)
        add_net = layers.flatten(add_net)
        add_net = layers.fully_connected(add_net, 512, normalizer_fn=layers.layer_norm)

        net = layers.linear(tf.concat([net, add_net], -1), 1)
        return net
