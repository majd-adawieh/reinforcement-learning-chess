from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Input, Model
import tensorflow as tf
from board import *


class A2C():
    def __init__(self):
        self.model = self.create_q_model()

    def create_q_model(self):
        input_layer = Input(shape=(8, 8, 16))
        x = Conv2D(filters=64, kernel_size=2, strides=(2, 2))(input_layer)
        x = Conv2D(filters=128, kernel_size=2, strides=(2, 2))(x)
        x = Conv2D(filters=256, kernel_size=2, strides=(2, 2))(x)
        x = Flatten()(x)
        action = Dense(4096, activation="softmax")(x)
        critic = Dense(1)(x)

        return Model(inputs=input_layer, outputs=[action, critic])
