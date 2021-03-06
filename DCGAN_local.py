# -*- coding: utf-8 -*-
"""### Setup environment"""

import tensorflow as tf
import keras
import glob
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import IPython.display as display
import os
from PIL import Image
from numpy.random import random
from tensorflow.keras import layers
import time
import imageio
import sys, getopt
# from contextlib import redirect_stdout

# Configure tensorflow
AUTOTUNE = tf.data.experimental.AUTOTUNE
#tf.enable_eager_execution()
tf.random.set_seed(0)

# declare parameters
BUFFER_SIZE = 400
BATCH_SIZE = 128
IMG_WIDTH = 64
IMG_HEIGHT = 64
noise_dim = 300
num_examples_to_generate = 25

"""## **1. Process Data**"""

#!unzip Hands.zip;
#!unzip arcDataset.zip;

"""### Helper fucntions for fetching and displaying images"""

# define function for loading images file
# input: file path
# output: eager tensor of shape=(IMG_WIDTH, IMG_HEIGHT, 3) of dtype=float32
def load(image_file):
  img = tf.io.read_file(image_file)
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  image = image[:, :, :]
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  #img = (img - 127.5) / 127.5 # Normalize the image to [-1, 1]
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# input: file path string
# output: decoded image
def process_path(file_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  print(img)
  img = decode_img(img)
  return img

# input: dataset
# output: shuffled and batched (size of BATCH_SIZE) dataset
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def show_batch(image_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.axis('off')

list_ds = tf.data.Dataset.list_files('Hands/*.jpg')

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
processed_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# shuffle data
train_ds = prepare_for_training(processed_ds, cache="./hands.tfcache")

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
init = RandomNormal(mean=0.0, stddev=0.02)

"""## **2. Define Models**

Adapted from:

https://www.tensorflow.org/tutorials/generative/dcgan

https://machinelearningmastery.com/how-to-get-started-with-generative-adversarial-networks-7-day-mini-course/

Convolutional Math:
http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html

### Generator

Dense layer: densely connected CNN

Upsample with transpose convolution layers.
"""

def make_generator_model():
    model = tf.keras.Sequential()
    # project and reshape
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size

    # upsample to 8x8x512
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

# Generate image from untrained generator
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0,:,:,0])

"""### Discriminator

Downsample with convolution layers.

Layers adapted from:
http://bamos.github.io/2016/08/09/deep-completion/
"""

def make_discriminator_model():
    model = tf.keras.Sequential()
    # downsample to 32x32x64
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # prevents overfitting

    # downsample to 16x16x128
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 16, 16, 128)

    # downsample to 8x8x256
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # downsample to 4x4xx512
    # can remove 1-2 layers for faster training but may have worse results
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # output to 1x1x1
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()
# decision = discriminator(generated_image)
# print (decision)

"""## **3. Define Loss and Optimizers**

Binary cross entropy is essentially the difference in the predicted probability of n and the actual probability of n.

Minimizing cross entropy is equivalent to minimizing the negative log likelihood of our data, which is a direct measure of the predictive power of our model.
"""

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# label smoothing 1 to [0.7,1.2]
def smooth_positive_labels(y):
    return y - 0.3 + (random(y.shape) * 0.5)

"""### Discriminator Loss

This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
"""

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(smooth_positive_labels(tf.ones_like(real_output)), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""### Generator Loss

The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

"""### Optimizers

Adam stochastic gradient descent computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
"""

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2,beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2,beta_1=0.5)

"""### Checkpoints"""

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints', max_to_keep=3)

"""## **4. Training**"""

"""### Training loop

The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
"""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(images):
    # random noise for generator
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      # calculate loss
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    # calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

"""### Training function

Times execution and displays the progress of the models by generating images from the seed. It saves a checkpoint every 5 epochs.
"""

def train(dataset, epochs,training):
  """Execute training loop on given dataset for n number of epochs.
     Generates GIF of progress at each epoch (synthesized images using the same seeded random input).

  Args:
    dataset:    dataset (tf.data.Dataset)
    epochs:     number of epochs to train for (int)
    training:   training session number (int)

  Returns:
    none

  """

  train_start = time.time()
  for epoch in range(epochs):
    start = time.time()
    batch_count = 0;

    for image_batch in dataset:
      (gen_loss, disc_loss) = train_step(image_batch)
      batch_count += 1
      display.clear_output(wait=True)
      print('Batch {0} processed with generator loss {1} and discriminator loss {2}'.format(batch_count, gen_loss, disc_loss))
      print('Epoch: {0}'.format(epoch + 1))

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed,
                             training)

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  end = time.time()
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed,
                           training)
  hours, rem = divmod(end-train_start, 3600)
  minutes, seconds = divmod(rem, 60)

  # save training time to txt
  save_models(training)

  print ('Time for training is {} hours and {} mins.'.format(int(hours),int(minutes)))

  # save training time to txt
  with open('train{:02d}/time.txt'.format(training), 'w') as f:
    f.write('Time for training is {} hours and {} mins.'.format(int(hours),int(minutes)))


"""### Restoring checkpoints and resuming training"""

def restore_train(dataset, epochs,training,offset):
  """Restore training loop from last checkpoint on given dataset for n number of epochs.
     Generates GIF of progress at each epoch (synthesized images using the same seeded random input).

  Args:
    dataset:    dataset (tf.data.Dataset)
    epochs:     number of epochs to train for (int)
    training:   training session number (int)
    offset:     epoch number of last checkpoint (int)

  Returns:
    none

  """
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    display.clear_output(wait=True)
    print("Restored from {} ending at step {}".format(manager.latest_checkpoint,int(checkpoint.step)))
  else:
    display.clear_output(wait=True)
    print("Initializing from scratch.")

  train_start = time.time()
  for epoch in range(offset,epochs+1):
    start = time.time()
    batch_count = 0;

    for image_batch in dataset:
      train_step(image_batch)
      batch_count += 1
      display.clear_output(wait=True)
      print('Batches processed {0}'.format(batch_count))
      print('Epoch: {0}'.format(epoch + 1))

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed,
                             training)

    # Save the model every _ epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  end = time.time()
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed,
                           training)
  hours, rem = divmod(end-train_start, 3600)
  minutes, seconds = divmod(rem, 60)

  save_models(training)

  print ('Time for training is {} hours and {} mins.'.format(int(hours),int(minutes)))

  # save training time to txt
  with open('train{:02d}/time.txt'.format(training), 'a') as f:
    f.write('Time for training is {} hours and {} mins.'.format(int(hours),int(minutes)))

"""### Helper functions to generate images and gifs"""

def generate_and_save_images(model, epoch, test_input,training):
  """Generates a batch of images at a given epoch.

  Args:
    model:          model object to generate with
    epoch:          epoch at which we have just finished training (int)
    training:       training session number (int)
    test_input:     latent input vector of data

  Returns:
    none

      """
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(10,10))

  for i in range(predictions.shape[0]):
      plt.subplot(5, 5, i+1)
      plt.imshow(predictions[i])# * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('train{:02d}/image_at_epoch_{:04d}.png'.format(training,epoch))

def generate_gif(training):
  anim_file = 'train{:02}/dcgan{:02d}.gif'.format(training,training)

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('train{:02}/image*.png'.format(training))
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
      frame = 2*(i**0.25)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

def save_models(training):
    generator.save(
        'train{:02d}/generator.h5'.format(training), overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None
    )
    discriminator.save(
        'train{:02d}/discriminator.h5'.format(training), overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None
    )

"""### Execution"""

def main(argv):
    """
	Usage:
				python DCGAN_local.py -r <offset> <training session #> <epochs>
				python DCGAN_local.py -t <training session #> <epochs>

    Options:
    -r          restore training starting from offset+1
	-t			start new training session

    Args:
    offset:		where the last session ended (int)
    training:	training session number (int)
    epochs:     number of epochs to run for (int)

    Returns:
    none

    """

    try:
        opts, args = getopt.getopt(argv,"r:t",["offset=","training=","epochs="])
    except getopt.GetoptError:
        print 'DCGAN_local.py [-r <offset>]/-t <training session #> <epochs>'

    training = int(args[0])
    EPOCHS = int(args[1])

    for opt, arg in opts:
        if opt in ("-r", "--offset"):
            restore_train(train_ds, EPOCHS, training, int(arg))
        else:
            os.mkdir('train{:02d}'.format(training)) # comment out if directory exists
            train(train_ds, EPOCHS,training)

    generate_gif(training)

    # def save_summary(training,model):
    #     with open('train{:02d}/{}summary.txt'.format(training,model), 'w') as f:
    #         with redirect_stdout(f):
    #             model.summary()
    #
    # save_summary(training,'generator')

if __name__== "__main__":
    main(sys.argv[1:])
