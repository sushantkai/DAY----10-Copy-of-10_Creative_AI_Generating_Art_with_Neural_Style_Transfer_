GAN Image Generation
Overview

This project shows how to build and train a Generative Adversarial Network (GAN) using TensorFlow to generate images. It works with datasets like CelebA or Oxford Flowers.

The project includes:

Loading and preprocessing images.

Building generator and discriminator models.

Defining loss functions and optimizers.

Training the GAN and updating weights in each step.

Generating and visualizing sample images during training.

Support for multi-GPU training with tf.distribute.MirroredStrategy.

How to Use

Install dependencies:

pip install tensorflow numpy matplotlib tqdm


Update DATA_PATH to your dataset folder.

Run the notebook step by step:

train(dataset, epochs=50)
generator.save('generator_model.h5')


Generate new images with the saved generator:

from tensorflow.keras.models import load_model
import tensorflow as tf

gen = load_model('generator_model.h5')
noise = tf.random.normal([16, 100])
images = gen(noise, training=False)

Goal

Learn how GANs work and create synthetic images. You can also modify it for tasks like image super-resolution or image-to-image translation.
