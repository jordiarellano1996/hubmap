"""
Standard Unet
"""
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Input, Dropout, \
    concatenate, Dense


class UnetBase:
    def conv_block(self, tensor, nfilters, size=3, drop_out=0.1, padding='same', activation="relu"):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, activation=activation, )(tensor)
        # x = Dropout(drop_out)(x)
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, activation=activation, )(x)
        return x

    def deconv_block(self, tensor, residual, nfilters, size=3, padding='same', strides=(2, 2), activation="relu",
                     drop_out=0.1):
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
        y = concatenate([y, residual])
        y = self.conv_block(y, nfilters, activation=activation, drop_out=drop_out)
        return y

    def get_layers(self):
        """ Create Unet model layers """
        return object, object


class Unet(UnetBase):
    def __init__(self, img_shape, n_classes=1, filters=64, drop_out=0.1):
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.filters = filters
        self.drop_out = drop_out

    def get_layers(self):
        # Unet inputs need to be divisible by 2^n, where n is the number of pool operators.
        # Input
        input_layer = Input(shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2]), name='image_input')
        # Down
        conv1 = self.conv_block(input_layer, size=3, nfilters=self.filters, drop_out=self.drop_out)
        conv1_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = self.conv_block(conv1_out, size=3, nfilters=self.filters * 2, drop_out=self.drop_out)
        conv2_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = self.conv_block(conv2_out, size=3, nfilters=self.filters * 4, drop_out=self.drop_out)
        conv3_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        conv4 = self.conv_block(conv3_out, size=3, nfilters=self.filters * 8, drop_out=self.drop_out)
        conv4_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        conv5 = self.conv_block(conv4_out, size=3, nfilters=self.filters * 16, drop_out=self.drop_out)

        # up
        deconv6 = self.deconv_block(conv5, residual=conv4, nfilters=self.filters * 8, drop_out=self.drop_out)
        deconv7 = self.deconv_block(deconv6, residual=conv3, nfilters=self.filters * 4, drop_out=self.drop_out)
        deconv8 = self.deconv_block(deconv7, residual=conv2, nfilters=self.filters * 2, drop_out=self.drop_out)
        deconv9 = self.deconv_block(deconv8, residual=conv1, nfilters=self.filters, drop_out=self.drop_out)

        # output
        output_layer = Conv2D(filters=self.n_classes, kernel_size=(1, 1), activation="sigmoid", padding='same')(deconv9)
        return input_layer, output_layer


class HalfUnet(UnetBase):
    def __init__(self, img_shape, n_classes=1, filters=64):
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.filters = filters

    def get_layers(self):
        # Input
        input_layer = Input(shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2]), name='image_input')
        # Down
        conv1 = self.conv_block(input_layer, size=3, nfilters=self.filters)
        conv1_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = self.conv_block(conv1_out, size=3, nfilters=self.filters)
        conv2_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = self.conv_block(conv2_out, size=3, nfilters=self.filters)
        conv3_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        conv4 = self.conv_block(conv3_out, size=3, nfilters=self.filters)
        conv4_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        conv5 = self.conv_block(conv4_out, size=3, nfilters=self.filters)

        # Up
        deconv6 = UpSampling2D(size=(2, 2))(conv5)
        deconv7 = UpSampling2D(size=(2, 2))(deconv6)
        deconv8 = UpSampling2D(size=(2, 2))(deconv7)
        deconv9 = UpSampling2D(size=(2, 2))(deconv8)
        conv6 = concatenate([deconv9, conv1])
        conv6_out = self.conv_block(conv6, size=3, nfilters=self.filters)

        # output
        output_layer = Conv2D(filters=self.n_classes, kernel_size=(1, 1), activation="sigmoid", padding='same')(
            conv6_out)

        return input_layer, output_layer


class Model:
    def __init__(self, input_layer: object, output_layer: object, loss: object, metrics: list,
                 learning_rate=0.001, verbose=False):
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.input_layer = input_layer
        self.output_layer = output_layer

    def get_model(self):
        model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)  # [self.input_layer]
        # model = PrintEveryBatch(model.input, model.output)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                            loss=self.loss, metrics=self.metrics, )

        if self.verbose:
            print(model.summary())

        return model


class PrintEveryBatch(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        tf.print(f"\n", output_stream=sys.stdout)
        tf.print("Input layer shape:", tf.shape(x), output_stream=sys.stdout)
        tf.print("Output layer shape:", tf.shape(y), output_stream=sys.stdout)
        tf.print("Input mean Tensorflow:", tf.reduce_mean(x), output_stream=sys.stdout)
        tf.print("Output mean Tensorflow:", tf.reduce_mean(x), output_stream=sys.stdout)
        return super().train_step(data)
