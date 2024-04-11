from tensorflow.keras import layers, Model, Input
from numpy import uint8
import keras_tuner as kt
import tensorflow as tf
import tensorflow_mri as tfmri
import utils.custom_losses as custom_losses

class DnCNN(kt.HyperModel):
  def __init__(self,
               config,
               config_model,
               input_shape,
               out_channels,
               loss,
               metrics,
               **kwargs):
    super().__init__(**kwargs)
    self.config=config
    self.config_model = config_model
    self.input_shape = input_shape
    self.out_channels = out_channels
    self.loss = loss
    self.metrics = metrics
    
  def build(self, hp):
    
    # Define hyperparameters
    residual_blocks = hp['residual_blocks']
    
    # Define model
    image_input = layers.Input(shape=(self.input_shape[0],self.input_shape[1],self.out_channels), name="image_input")
    residual_input = layers.Input(shape=(self.input_shape[0],self.input_shape[1],self.out_channels), name="residual_input")
    
    x = ReflectionPadding2D(padding=(1,1))(image_input)
    x = layers.Conv2D(64, 3, padding="valid")(x)
    x = layers.ReLU()(x)

    for i in range(residual_blocks):
      x = ResidualBlock()(x)

    x = ReflectionPadding2D(padding=(1,1))(x)
    x = layers.Conv2D(64, 3, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = ReflectionPadding2D(padding=(1,1))(x)
    residual_output = layers.Conv2D(self.out_channels, 3, padding="valid", name="residual_output")(x)

    image_output = layers.Add(name="image_output")([residual_output, image_input])
    
    model = Model(inputs=[image_input, residual_input], outputs=[image_output, residual_output])
    
    # Add Jensen Shannon loss
    JS_residual_loss = custom_losses.jensen_shannon_loss(residual_input, residual_output)
    model.add_loss(self.config_model['JS_lambda']*JS_residual_loss)
    model.add_metric(JS_residual_loss, name = 'JS_residual_loss', aggregation = 'mean')
    
    # Compile model
    model.compile(loss={"image_output":self.loss},
                  optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
                  metrics=self.metrics)
    
    return model
  
  def fit(self, hp, model, train_ds, val_ds, **kwargs):
    # Define the number of epochs
    if 'epochs' in kwargs:
      initial_epoch = 0
      last_epoch = kwargs['epochs']
    else:
      initial_epoch = 0
      last_epoch = 100
      
    # Fit the model
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        initial_epoch=initial_epoch,
                        epochs=last_epoch,
                        verbose=1,
                        # callbacks=kwargs['callbacks'],
    )
    
    return history
    




def ResidualDnCNN(residual_blocks=9, out_channels=1, image_shape=[80,80], JS_lambda=0.05):
  
  image_shape = uint8(image_shape)
  image_input = layers.Input(shape=(image_shape[0],image_shape[1],out_channels), name="image_input")
  residual_input = layers.Input(shape=(image_shape[0],image_shape[1],out_channels), name="residual_input")

  x = ReflectionPadding2D(padding=(1,1))(image_input)
  x = layers.Conv2D(64, 3, padding="valid")(x)
  x = layers.ReLU()(x)

  for i in range(residual_blocks):
    x = ResidualBlock()(x)

  x = ReflectionPadding2D(padding=(1,1))(x)
  x = layers.Conv2D(64, 3, padding="valid")(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  
  x = ReflectionPadding2D(padding=(1,1))(x)
  residual_output = layers.Conv2D(out_channels, 3, padding="valid", name="residual_output")(x)

  # Global residual
  image_output = layers.Add(name="image_output")([residual_output, image_input])

  model = Model(inputs=[image_input, residual_input], outputs=[image_output, residual_output])
  
  # Jensen Shannon loss
  JS_residual_loss = custom_losses.jensen_shannon_loss(residual_input, residual_output)
  model.add_loss(JS_lambda*JS_residual_loss)
  model.add_metric(JS_residual_loss, name = 'JS_residual_loss', aggregation = 'mean')
  
  return model


# Residual block
@tf.keras.utils.register_keras_serializable()
class ResidualBlock(layers.Layer):
  
  def __init__(self, **kwargs):
    super(ResidualBlock, self).__init__()
    
    # First block
    self.reflection_padding1 = ReflectionPadding2D(padding=(1,1))
    self.conv1 = layers.Conv2D(64, 3, padding="valid")
    self.batch_normalization1 = layers.BatchNormalization()
    self.ReLU1 = layers.ReLU()
    
    # Second block
    self.reflection_padding2 = ReflectionPadding2D(padding=(1,1))
    self.conv2 = layers.Conv2D(64, 3, padding="valid")
    self.batch_normalization2 = layers.BatchNormalization()
    self.ReLU2 = layers.ReLU()
  
  def call(self, inputs):
    x = self.reflection_padding1(inputs)
    x = self.conv1(x)
    x = self.batch_normalization1(x)
    x = self.ReLU1(x)
    
    x = self.reflection_padding2(x)
    x = self.conv2(x)
    x = self.batch_normalization2(x)
    x = self.ReLU2(x)
    
    # Internal residual
    x = layers.Add()([x, inputs])
    
    return x


# Reflection padding layer
@tf.keras.utils.register_keras_serializable()
class ReflectionPadding2D(layers.Layer):
    
  def __init__(self, padding=(1,1), **kwargs):
    self.padding = tuple(padding)
    super(ReflectionPadding2D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
  
  def get_config(self):
    config = super().get_config()
    config.update({"padding": self.padding})
    return config

  def call(self, input_tensor, mask=None):
    padding_width, padding_height = self.padding
    return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')
      