import pydicom as dcm
import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmri
from scipy.ndimage import gaussian_filter

# Function to prepare fastMRI data for training (resizing image, adding high intensity circle to mimick 100 mM reference phantom)
def prepare_data(file, cohort, image_shape):
    
    # Load dicom image
    data = dcm.dcmread(file + '.dcm')
    image = data.pixel_array

    # Crop
    cropped_matrix = [np.max(image.shape), np.max(image.shape)]
    cropped_image = tf.dtypes.complex(tfmri.resize_with_crop_or_pad(tf.convert_to_tensor(image, dtype=tf.dtypes.float32), cropped_matrix, padding_mode='reflect'), tf.zeros(cropped_matrix, dtype=tf.dtypes.float32))

    # Downsample
    new_matrix = tf.convert_to_tensor(np.array(image_shape), dtype=tf.int32)
    cropped_kspace = tfmri.signal.fft(cropped_image, shift='True')
    low_res_kspace = tfmri.resize_with_crop_or_pad(cropped_kspace, new_matrix)
    low_res_image = tfmri.signal.ifft(low_res_kspace, shift='True')

    # Add 'phantom'
    low_res_image = add_phantom(low_res_image, radius=6, scale=7.5)
 
    # Save prepared image as .npy array
    np.save(f"data/knee_mri_clinical_seq_batch2/{cohort}/{file.split('/')[-1]}.npy", low_res_image)

# Function to add high signal intensity circle to corner of image, mimicking 100 mM sodium reference phantom
def add_phantom(image, radius, scale):
    h = image.shape[0]
    w = image.shape[1]
    mask = create_circular_mask(h, w, radius)
    inverted_mask = 1 - mask
    intensity = np.mean(image) * scale
    phantom_image = (image * inverted_mask) + (mask * intensity)
    return phantom_image

def create_circular_mask(h, w, radius):
    center = random_circle_location(h, w, radius)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.array(dist_from_center <= radius, dtype=float)
    filtered_mask = gaussian_filter(mask, 0.6)
    return filtered_mask

def random_circle_location(h, w, radius):
    corner = np.random.randint(0, 4)
    edge = 3
    shift = np.random.randint(0, 6, size=[2,1])
    if corner == 0:
        x = radius + edge + shift[0]
        y = radius + edge + shift[1]
    elif corner == 1:
        x = radius + edge + shift[0]
        y = h - radius - edge - shift[1]
    elif corner == 2:
        x = w - radius - edge - shift[0]
        y = h - radius - edge - shift[1]
    elif corner == 3:
        x = w - radius - edge - shift[0]
        y = radius + edge + shift[1]
    return [x, y]