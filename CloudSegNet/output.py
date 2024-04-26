import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread

import tensorflow as tf

# Load the trained autoencoder model using TensorFlow's keras module
autoencoder = tf.keras.models.load_model('./results/balanced_random_sample/' + 'cloudsegnet.hdf5')  # Replace 'your_model.h5' with the path to your HDF5 model file

# Path to the folder containing test images
test_images_folder = './dataset/testimages'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Load test images
test_images = [imread(os.path.join(test_images_folder, f)) for f in image_files]

# Convert the list of images to a numpy array
test_images = np.array(test_images)

# Check if the images are grayscale or RGB
if len(test_images.shape) == 3:
    # Grayscale images
    height, width, channels = test_images.shape
    test_images = np.expand_dims(test_images, axis=-1)  # Add channel dimension
else:
    # RGB images
    height, width, channels = test_images.shape[1:]

# Predict segmentation masks for test images
segmentation_masks = autoencoder.predict(test_images)

# Visualize results
for i in range(len(test_images)):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(test_images[i].squeeze(), cmap='gray' if channels == 1 else None)
    plt.title('Original Image')

    # Segmentation Mask
    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_masks[i][:, :, 0], cmap='gray')  # Assuming the output is a single-channel mask
    plt.title('Segmentation Mask')

    # Reconstructed Image (optional)
    reconstructed_img = autoencoder.predict(np.expand_dims(test_images[i], axis=0))
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_img[0].squeeze(), cmap='gray' if channels == 1 else None)
    plt.title('Reconstructed Image')

    plt.show()
