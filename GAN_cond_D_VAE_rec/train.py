import tensorflow as tf
import networks
# import util
from tensorflow import flags
import tensorflow.contrib.gan as tfgan
from gan_tf_examples.mnist.data_provider import provide_data
from VAE.variational_autoencoder import VAE


flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', 'mnist_norm_images',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', "../data/mnist", 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string(
    'gan_type', 'unconditional',
    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer(
    'grid_size', 5, 'Grid size for image visualization.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS


def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, one_hot_labels, _ = provide_data('train', FLAGS.batch_size, FLAGS.dataset_dir, num_threads=4)
            images = 2.0 * images - 1.0

    rec = None

    gan_model = tfgan.gan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        real_data=images,
        generator_inputs=[tf.random_normal(
            [FLAGS.batch_size, FLAGS.noise_dims]), rec])

    tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size, False)

    # Get the GANLoss tuple. You can pass a custom function, use one of the
    # already-implemented losses from the losses library, or use the defaults.
    with tf.name_scope('loss'):

        gan_loss = tfgan.gan_loss(
            gan_model,
            gradient_penalty_weight=1.0,
            mutual_information_penalty_weight=0.0,
            add_summaries=True)
        # tfgan.eval.add_regularization_loss_summaries(gan_model)

    # Get the GANTrain ops using custom optimizers.
    with tf.name_scope('train'):
        gen_lr, dis_lr = (1e-3, 1e-4)
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
            discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
            summarize_gradients=False,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0:
        return
    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        logdir=FLAGS.train_log_dir,
        save_summaries_steps=500,
        get_hooks_fn=tfgan.get_joint_train_hooks())


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
