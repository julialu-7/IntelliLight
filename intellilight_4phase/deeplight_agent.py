"""
deeplight_agent.py

Deep reinforcement learning traffic signal control agent. Implements a phase-gated Q-learning model using Keras, with options for
separate experience replay buffers per (phase, action) pair.
"""

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add, concatenate, add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
import random
import os

from network_agent import NetworkAgent, conv2d_bn, Selector, State

MEMO = "Deeplight"


class DeeplightAgent(NetworkAgent):
    """
    Deep Q-network-based agent for adaptive traffic light control with phase-specific gating.
    """

    def __init__(self, num_phases, num_actions, path_set):
        """
        Initializes the DeeplightAgent with network and memory buffers.
        """
        super(DeeplightAgent, self).__init__(num_phases=num_phases, path_set=path_set)
        self.num_actions = num_actions

        # Main Q-network
        self.q_network = self.build_network()
        self.save_model("init_model")
        self.update_outdated = 0

        # Target Q-network for stable training
        self.q_network_bar = self.build_network_from_copy(self.q_network)
        self.q_bar_outdated = 0

        # Experience replay buffer
        if not self.para_set.SEPARATE_MEMORY:
            self.memory = self.build_memory()
        else:
            self.memory = self.build_memory_separate()

        self.average_reward = None

    def reset_update_count(self):
        """Reset the update counters for model and target network."""
        self.update_outdated = 0
        self.q_bar_outdated = 0

    def set_update_outdated(self):
        """Force update flags to trigger training and target sync."""
        self.update_outdated = -2 * self.para_set.UPDATE_PERIOD
        self.q_bar_outdated = 2 * self.para_set.UPDATE_Q_BAR_FREQ

    def convert_state_to_input(self, state):
        """Convert State object to a list of input tensors for the network."""
        return [getattr(state, feature_name) for feature_name in self.para_set.LIST_STATE_FEATURE]

    def build_network(self):
        """Construct the Q-network with optional phase gating."""
        dic_input_node = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            shape = getattr(State, "D_" + feature_name.upper())
            dic_input_node[feature_name] = Input(shape=shape, name="input_" + feature_name)

        # Apply CNN to image-like features, flatten others
        dic_flatten_node = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            shape = getattr(State, "D_" + feature_name.upper())
            if len(shape) > 1:
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        # Concatenate all features into one vector
        list_all_flatten_feature = [dic_flatten_node[feature] for feature in self.para_set.LIST_STATE_FEATURE]
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # Apply shared dense layers to concatenated features
        shared_dense = self._shared_network_structure(all_flatten_feature, self.para_set.D_DENSE)

        # Phase selector logic: each phase has a separate Q-value branch
        if "cur_phase" in self.para_set.LIST_STATE_FEATURE and self.para_set.PHASE_SELECTOR:
            list_selected_q_values = []
            for phase in range(self.num_phases):
                q_values = self._separate_network_structure(shared_dense, self.para_set.D_DENSE, self.num_actions, memo=phase)
                selector = Selector(phase, name=f"selector_{phase}")(dic_input_node["cur_phase"])
                selected_q_values = Multiply(name=f"multiply_{phase}")([q_values, selector])
                list_selected_q_values.append(selected_q_values)
            q_values = Add()(list_selected_q_values)
        else:
            q_values = self._separate_network_structure(shared_dense, self.para_set.D_DENSE, self.num_actions)

        # Build and compile the model
        network = Model(inputs=[dic_input_node[feature_name] for feature_name in self.para_set.LIST_STATE_FEATURE],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(learning_rate=self.para_set.LEARNING_RATE),
                        loss="mean_squared_error")
        network.summary()
        return network

    def build_memory_separate(self):
        """Initialize memory as a list."""
        memory_list=[]
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember(self, state, action, reward, next_state):
        """Store experience tuple into memory."""
        if self.para_set.SEPARATE_MEMORY:
            ''' log the history separately '''
            self.memory[state.cur_phase[0][0]][action].append([state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain):
        """Trim memory to maintain size limits."""
        if self.para_set.SEPARATE_MEMORY:
            # remove the old history if the memory is too large, in a separate way
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[phase_i][action_i])
                    if len(self.memory[phase_i][action_i]) > self.para_set.MAX_MEMORY_LEN:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[phase_i][action_i])))
                        self.memory[phase_i][action_i] = self.memory[phase_i][action_i][-self.para_set.MAX_MEMORY_LEN:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.memory[phase_i][action_i])))
        else:
            if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
                print("length of memory: {0}, before forget".format(len(self.memory)))
                self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

    def _cal_average(self, sample_memory):
        """Compute average reward per phase-action pair from flat memory."""
        list_reward = []
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            list_reward.append([])
            for action_i in range(self.num_actions):
                list_reward[phase_i].append([])
        for [state, action, reward, _] in sample_memory:
            phase = state.cur_phase[0][0]
            list_reward[phase][action].append(reward)

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(list_reward[phase_i][action_i]) != 0:
                    average_reward[phase_i][action_i] = np.average(list_reward[phase_i][action_i])

        return average_reward

    def _cal_average_separate(self,sample_memory):
        """Compute average reward from separate memory per phase/action."""
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                len_sample_memory = len(sample_memory[phase_i][action_i])
                if len_sample_memory > 0:
                    list_reward = []
                    for i in range(len_sample_memory):
                        state, action, reward, _ = sample_memory[phase_i][action_i][i]
                        list_reward.append(reward)
                    average_reward[phase_i][action_i]=np.average(list_reward)
        return average_reward

    def get_sample(self, memory_slice, dic_state_feature_arrays, Y, gamma, prefix, use_average):
        """Convert sampled memory into training data arrays X and Y."""
        len_memory_slice = len(memory_slice)
        f_samples = open(os.path.join(self.path_set.PATH_TO_OUTPUT, "{0}_memory".format(prefix)), "a")
        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]
            for feature_name in self.para_set.LIST_STATE_FEATURE:
                dic_state_feature_arrays[feature_name].append(getattr(state, feature_name)[0])
            if state.if_terminal:
                next_estimated_reward = 0
            else:
                next_estimated_reward = self._get_next_estimated_reward(next_state)
            total_reward = reward + gamma * next_estimated_reward
            if not use_average:
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
            else:
                target = np.copy(np.array([self.average_reward[state.cur_phase[0][0]]]))

            pre_target = np.copy(target)
            target[0][action] = total_reward
            Y.append(target[0])

            for feature_name in self.para_set.LIST_STATE_FEATURE:
                if "map" not in feature_name:
                    f_samples.write("{0}\t".format(str(getattr(state, feature_name))))
            f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                str(pre_target), str(target),
                str(action), str(reward), str(next_estimated_reward)
            ))
        f_samples.close()

        return dic_state_feature_arrays, Y

    def train_network(self, Xs, Y, prefix, if_pretrain):
        """Train Q-network on the sampled (X, Y) data."""
        if if_pretrain:
            epochs = self.para_set.EPOCHS_PRETRAIN
        else:
            epochs = self.para_set.EPOCHS
        batch_size = min(self.para_set.BATCH_SIZE, len(Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.para_set.PATIENCE, verbose=0, mode='min')

        hist = self.q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3, callbacks=[early_stopping])
        self.save_model(prefix)

    def update_network(self, if_pretrain, use_average, current_time):
        """Trigger Q-network training based on update period and collect training samples."""
        if current_time - self.update_outdated < self.para_set.UPDATE_PERIOD:
            return

        self.update_outdated = current_time

        # prepare the samples
        if if_pretrain:
            gamma = self.para_set.GAMMA_PRETRAIN
        else:
            gamma = self.para_set.GAMMA

        dic_state_feature_arrays = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        # get average state-action reward
        if self.para_set.SEPARATE_MEMORY:
            self.average_reward = self._cal_average_separate(self.memory)
        else:
            self.average_reward = self._cal_average(self.memory)

        # ================ sample memory ====================
        if self.para_set.SEPARATE_MEMORY:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=self.para_set.PRIORITY_SAMPLING,
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    dic_state_feature_arrays, Y = self.get_sample(
                        sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=self.para_set.PRIORITY_SAMPLING,
                memory=self.memory,
                if_pretrain=if_pretrain)
            dic_state_feature_arrays, Y = self.get_sample(
                sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        # ================ sample memory ====================

        Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in self.para_set.LIST_STATE_FEATURE]
        Y = np.array(Y)
        sample_weight = np.ones(len(Y))
        # shuffle the training samples, especially for different phases and actions
        Xs, Y, _ = self._unison_shuffled_copies(Xs, Y, sample_weight)

        # ============================  training  =======================================

        self.train_network(Xs, Y, current_time, if_pretrain)
        self.q_bar_outdated += 1
        self.forget(if_pretrain=if_pretrain)

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain):
        """Sample memory with or without priority-based weighting."""
        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(self.para_set.SAMPLE_SIZE, len_memory)
        else:
            sample_size = min(self.para_set.SAMPLE_SIZE_PRETRAIN, len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]

                if state.if_terminal:
                    next_estimated_reward = 0
                else:
                    next_estimated_reward = self._get_next_estimated_reward(next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)), weights=priority, k=sample_size)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    @staticmethod
    def _cal_priority(sample_weight):
        """Calculate normalized priority sampling weights."""
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np
