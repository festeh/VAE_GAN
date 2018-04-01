import tensorflow as tf
from GAN_cond_G_D_vae_gen import networks
from gan_tf_examples.mnist import util
from tensorflow import flags
import tensorflow.contrib.gan as tfgan
from gan_tf_examples.mnist.data_provider import provide_data
from VAE.variational_autoencoder import VAE


flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', 'lul',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', "../data/mnist", 'Location of data.')

flags.DEFINE_string('vae_checkpoint_folder', None, 'Location of the saved VAE model')



flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string(
    'gan_type', 'unconditional',
    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer(
    'grid_size', 8, 'Grid size for image visualization.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

flags.DEFINE_integer(
    'num_images_eval', 10000, 'How many generated images use in evaluation')

FLAGS = flags.FLAGS
MNIST_CLASSIFIER_FROZEN_GRAPH = '../gan_tf_examples/mnist/data/classify_mnist_graph_def.pb'

def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images_vae, one_hot_labels, _ = provide_data('train', FLAGS.batch_size, FLAGS.dataset_dir, num_threads=4)
            images_gan = 2.0 * images_vae - 1.0

    my_vae = VAE("train", z_dim=64, data_tensor=images_vae)
    vae_checkpoint_path = tf.train.latest_checkpoint(FLAGS.vae_checkpoint_folder)
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])
    gen_images = 2.0 * my_vae.Decoder(noise)[1] - 1
    rec_images = 2.0 * my_vae.reconstruct(images_vae) - 1
    saver = tf.train.Saver()

    gan_model = tfgan.gan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        real_data=tf.stack([images_gan, tf.reshape(rec_images, [FLAGS.batch_size, 28, 28, 1])]),
        generator_inputs=[noise, tf.reshape(gen_images, [FLAGS.batch_size, 28, 28, 1])],
        check_shapes=False)

    add_noise = tf.random_normal([FLAGS.num_images_eval, FLAGS.noise_dims])
    add_gen_images = my_vae.Decoder(add_noise)[1]
    with tf.variable_scope('Generator', reuse=True):
        eval_images = gan_model.generator_fn(
            [add_noise, tf.reshape(add_gen_images, [FLAGS.num_images_eval, 28, 28, 1])],
            is_training=False)

    # Calculate Inception score.
    tf.summary.scalar("Inception score", util.mnist_score(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH))

    # Calculate Frechet Inception distance.
    with tf.device('/cpu:0'):
        real_images, labels, _ = provide_data(
            'train', FLAGS.num_images_eval, FLAGS.dataset_dir)
    tf.summary.scalar("Frechet distance", util.mnist_frechet_distance(
        real_images, eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH))

    tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size, True)

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

    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

    step_hooks = tfgan.get_sequential_train_hooks()(train_ops)
    hooks = [tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
             tf.train.LoggingTensorHook([status_message], every_n_iter=10)] + list(step_hooks)

    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           save_summaries_steps=500,
                                           checkpoint_dir=FLAGS.train_log_dir) as sess:
        saver.restore(sess, vae_checkpoint_path)
        loss = None
        while not sess.should_stop():
            loss = sess.run(train_ops.global_step_inc_op)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
