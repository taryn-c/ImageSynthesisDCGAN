import tensorflow as tf
import keras
import math
import sys
from matplotlib import pyplot as plt
from numpy.random import random

def generate_and_save_images(model,test_input,training,num):

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
	num_subplots = math.ceil(math.sqrt(num))

	for i in range(predictions.shape[0]):
		plt.subplot(num_subplots, num_subplots, i+1)
		plt.imshow(predictions[i])# * 127.5 + 127.5)
		plt.axis('off')

	plt.tight_layout()
	plt.show()

def main(argv):
    """Usage: python generate_image.py <training> <noise_dim> <num>

    Args:
    training:       training session number (int)
    noise_dim:      noise dimension that model uses
    num:            number of examples to generate

    Returns:
    none

    """

    training = int(argv[0])
    noise_dim = int(argv[1])
    num_examples_to_generate = int(argv[2])

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    generator = tf.keras.models.load_model('train{:02d}/generator.h5'.format(training))

    generate_and_save_images(generator,seed,training,num_examples_to_generate)

if __name__== "__main__":
    main(sys.argv[1:])
