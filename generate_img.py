import tensorflow as tf
import keras
import math
import sys
from matplotlib import pyplot as plt
from numpy.random import random
import os

def generate_and_save_images(model,test_input,training,grid_size,num):

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

	fig = plt.figure()#figsize=(10,10))
	num_subplots = math.ceil(math.sqrt(grid_size))

	for i in range(predictions.shape[0]):
		plt.subplot(num_subplots, num_subplots, i+1)
		plt.imshow(predictions[i])# * 127.5 + 127.5)
		plt.axis('off')

	plt.tight_layout()
	#plt.show()
	plt.savefig('train{:02d}/images/image{:04d}.png'.format(training,num))

def main(argv):
	"""Usage: python generate_img.py <training> <noise_dim> <grid> <num>

	Args:
	training:       training session number (int)
	noise_dim:      noise dimension that model uses
	grid:			size of grid
	num:            number of images to generate

	Returns:
	none

	"""

	training = int(argv[0])
	noise_dim = int(argv[1])
	grid_size = int(argv[2])

	generator = tf.keras.models.load_model('train{:02d}/generator.h5'.format(training))
	if not os.path.isdir('train{:02d}/images'.format(training)):
		os.mkdir('train{:02d}/images'.format(training))
		
	for img in range((int(argv[3])+1)):
		seed = tf.random.normal([grid_size, noise_dim])
		generate_and_save_images(generator,seed,training,grid_size,img)

if __name__== "__main__":
    main(sys.argv[1:])
