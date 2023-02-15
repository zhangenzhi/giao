import tensorflow as tf

class UNet:
    def __init__(self, input_shape=[128, 128, 3]) -> None:
        from tensorflow_examples.models.pix2pix import pix2pix
        OUTPUT_CLASSES = input_shape[-1]
    
        self.input_shape = input_shape
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        self.down_stack.trainable = False

        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        self.model = self.unet_model(output_channels=OUTPUT_CLASSES)

    def unet_model(self, output_channels:int):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Downsampling through the model
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def plot_model(self):
        tf.keras.utils.plot_model(self.model, show_shapes=True)
    
    def __call__(self, inputs):
        return self.model(inputs)
    
def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
    
class CUNet:
    
    def __init__(self, input_shape=[32,32,3]) -> None:
        self.input_shape = input_shape
        self.OUTPUT_CHANNELS = input_shape[-1]
        self.latent_phase = None
        self.model = self.Generator()
    
    def Generator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        down_stack = [
            downsample(32, 4, apply_batchnorm=False),  # (batch_size, 16, 16, 64)
            downsample(64, 4),  # (batch_size, 8, 8, 128)
            downsample(128, 4),  # (batch_size, 4, 4, 256)
            downsample(256, 4),  # (batch_size, 2, 2, 512)
            # downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(128, 4),  # (batch_size, 8, 8, 512)
            upsample(64, 4),  # (batch_size, 16, 16, 256)
            upsample(32, 4),  # (batch_size, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 32, 32, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        latent_phase = skips[0]
        
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=(x,latent_phase))
    
    def plot_model(self):
        tf.keras.utils.plot_model(self.model, show_shapes=True)
    
    def __call__(self, inputs):
        return self.model(inputs)