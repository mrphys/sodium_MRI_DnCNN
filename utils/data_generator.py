import tensorflow as tf
import tensorflow_mri as tfmri
import numpy as np
# from tensorflow.keras import layers

# Data generator
class CustomDataGen():    
  def __init__(self, 
              patients,
              cohort,
              traj_config,
              NSA=30,
              noise_level=0.038,
              image_shape=[80,80]
              ):
    # random.shuffle(patients)
    self.patients = patients
    self.cohort = cohort
    self.averages = NSA
    self.noise_level = noise_level
    self.image_shape = image_shape
    
    # Determine spiral trajectory, image shape and density weights from sodium dataset
    # self.traj = recon_utils.calculate_trajectory(twixObj, data_hdr, sodium_image_shape, image_dim)
    self.traj = tf.reshape(tfmri.sampling.spiral_trajectory(**traj_config), [-1,2])
    self.dens = tfmri.sampling.estimate_density(self.traj, self.image_shape, method="pipe")
    self.out_channels = 1
    self.input_shape = self.output_shape = [self.image_shape[0], self.image_shape[1], self.out_channels]
    self.output_signature = ({'image_input': tf.TensorSpec(shape=self.input_shape, dtype=tf.float32), 'residual_input':tf.TensorSpec(shape=self.input_shape, dtype=tf.float32)}, 
                    tf.TensorSpec(shape=self.output_shape, dtype=tf.float32))
            
  def data_generator(self):
    for patient in self.patients:
      # Load low res images
      low_res_image = np.load(f"data/knee_mri_clinical_seq_batch2/{self.cohort}/{patient.split('/')[-1]}.npy")
      low_res_image = low_res_image*(1j)
      
      # Resample spiral kspace
      spiral_kspace = tfmri.signal.nufft(low_res_image, self.traj,
                                          transform_type='type_2',
                                          fft_direction='forward',
                                          tol=1e-14)
      
      # Reconstruct clean spiral image
      clean_image = tfmri.recon.adjoint(spiral_kspace,
                                        self.image_shape,
                                        trajectory=self.traj,
                                        density=self.dens)
      
      # Determine simulated noise level
      kspace_max = np.max(np.abs(spiral_kspace))
      sigma = kspace_max * self.noise_level
      
      # Add noise to 150 averages
      temp = []
      for i in range(150):
          temp.append(spiral_kspace + np.random.normal(0, sigma, spiral_kspace.shape) + np.random.normal(0, sigma, spiral_kspace.shape)*1j)
      spiral_kspace = np.array(temp)

      # Reconstruct with reduced averages
      mean_kspace = np.mean(spiral_kspace[0:self.averages,:], axis=-2)
      noise_image = tfmri.recon.adjoint(mean_kspace,
                                        self.image_shape,
                                        trajectory=self.traj,
                                        density=self.dens)
      
      # MAGNTIDUE INPUT
      shift, scale = normalize_var(noise_image)
      noise_slice = normalize(tf.expand_dims(noise_image, -1), shift, scale)
      clean_slice = normalize(tf.expand_dims(clean_image, -1), shift, scale)
      
      residuals = clean_slice - noise_slice
      
      yield {'image_input':tf.math.abs(noise_slice), 'residual_input':tf.math.abs(residuals)}, tf.math.abs(clean_slice)
          
  def get_gen(self):
    return self.data_generator()
  
  def get_output_signature(self):
    return self.output_signature
  
  def get_out_channels(self):
    return self.out_channels

# Normalize functions
def normalize_var(image, shift=None, scale=None):
  if shift ==  None:
    shift = np.amin(tf.math.abs(image))
  temp_image = image - shift
  if scale == None:
    scale = np.amax(tf.math.abs(temp_image))
  return shift, scale

def normalize(image, shift=None, scale=None):
  if shift == None and scale == None:
    temp_image = image - np.amin(tf.math.abs(image))
    return temp_image/np.amax(tf.math.abs(temp_image))
  else:
    return (image-shift)/scale