from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Input, Model
import tensorflow as tf
from board import *


class Q_model():
    def __init__(self):
        self.model = self.create_q_model()
        self.agent_num = 0

    def change_agent(self):
        if (self.agent_num == 0):
            self.agent_num = 1
        else:
            self.agent_num = 0

    def reset_agent_num(self):
        self.agent_num = 0

    def create_q_model(self):
        input_layer = Input(shape=(8, 8, 12))
        x = Conv2D(filters=64, kernel_size=2, strides=(2, 2))(input_layer)
        x = Conv2D(filters=128, kernel_size=2, strides=(2, 2))(x)
        x = Conv2D(filters=256, kernel_size=2, strides=(2, 2))(x)
        x = Flatten()(x)

        action = Dense(4096, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=action)

    def predict(self, env):
        state_tensor = tf.convert_to_tensor(env.translated_board)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action_space = filter_legal_moves(
            env.board, action_probs[0], self.agent_num, env.translated_board)
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move, action

    def explore(self, env):
        action_space = 1 * 2 + np.random.uniform(0, 1, 4096)
        action_space = filter_legal_moves(
            env.board, action_space, self.agent_num, env.translated_board)
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move, action
