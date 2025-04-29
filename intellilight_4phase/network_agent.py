"""
network_agent.py

Extension of the base Agent class, implementing a deep Q-network agent
with convolutional feature extraction and separate Q-branches for each phase.
"""

import numpy as np
from keras.layers import (Input, Dense, Conv2D, Flatten, BatchNormalization,
                           Activation, Multiply, Add, Dropout, MaxPooling2D, Layer)
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras import backend as K
import random
import os

from agent import Agent, State


class Selector(Layer):
    """Custom Keras layer that selects phase-specific outputs."""

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        """Build the selector layer without trainable parameters."""
        super(Selector, self).build(input_shape)

    def call(self, x):
        """Apply phase-based selection mask."""
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        """Return layer configuration."""
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """Return output shape (same as input)."""
        return input_shape


def conv2d_bn(input_layer, index_layer, filters=16, kernel_size=(3, 3), strides=(1, 1)):
    """Apply Conv2D + BatchNormalization + Activation + Pooling + Dropout."""
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                  padding='same', use_bias=False, name=f"conv{index_layer}")(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name=f"bn{index_layer}")(conv)
    act = Activation('relu', name=f"act{index_layer}")(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x


class NetworkAgent(Agent):
    """Deep Q-learning agent with convolutional feature extraction."""

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        """Shuffle X, Y, and sample weights consistently."""
        p = np.random.permutation(len(Y))
        new_Xs = [x[p] for x in Xs]
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        """Construct CNN feature extractor network."""
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        """Build shared dense layer for feature extraction."""
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        """Create phase-specific dense branches for Q-value prediction."""
        hidden_1 = Dense(dense_d, activation="sigmoid", name=f"hidden_separate_branch_{memo}_1")(state_features)
        q_values = Dense(num_actions, activation="linear", name=f"q_values_separate_branch_{memo}")(hidden_1)
        return q_values

    def load_model(self, file_name):
        """Load a saved Q-network model from file."""
        self.q_network = load_model(os.path.join(self.path_set.PATH_TO_MODEL, f"{file_name}_q_network.h5"))

    def save_model(self, file_name):
        """Save current Q-network model to file."""
        self.q_network.save(os.path.join(self.path_set.PATH_TO_MODEL, f"{file_name}_q_network.h5"))

    def choose(self, count, if_pretrain):
        """Select an action using epsilon-greedy strategy."""
        q_values = self.q_network.predict(self.convert_state_to_input(self.state))
        if if_pretrain:
            self.action = np.argmax(q_values[0])
        else:
            if random.random() <= self.para_set.EPSILON:
                self.action = random.randrange(len(q_values[0]))
                print("##Explore")
            else:
                self.action = np.argmax(q_values[0])
            if self.para_set.EPSILON > 0.001 and count >= 20000:
                self.para_set.EPSILON *= 0.9999
        return self.action, q_values

    def build_memory(self):
        """Initialize empty memory buffer."""
        return []

    def build_network_from_copy(self, network_copy):
        """Rebuild a Q-network from a saved network copy."""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(learning_rate=self.para_set.LEARNING_RATE),
                        loss="mean_squared_error")
        return network

    def remember(self, state, action, reward, next_state):
        """Log a transition (state, action, reward, next_state) into memory."""
        self.memory.append([state, action, reward, next_state])

    def forget(self):
        """Trim the oldest entries if memory exceeds maximum size."""
        if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
            print(f"length of memory: {len(self.memory)}, before forget")
            self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            print(f"length of memory: {len(self.memory)}, after forget")

    def _get_next_estimated_reward(self, next_state):
        """Estimate Q-value for the next state."""
        if self.para_set.DDQN:
            a_max = np.argmax(self.q_network.predict(self.convert_state_to_input(next_state))[0])
            next_estimated_reward = self.q_network_bar.predict(self.convert_state_to_input(next_state))[0][a_max]
        else:
            next_estimated_reward = np.max(self.q_network_bar.predict(self.convert_state_to_input(next_state))[0])
        return next_estimated_reward

    def update_network_bar(self):
        """Periodically update the target Q-network (Q-bar)."""
        if self.q_bar_outdated >= self.para_set.UPDATE_Q_BAR_FREQ:
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.q_bar_outdated = 0
