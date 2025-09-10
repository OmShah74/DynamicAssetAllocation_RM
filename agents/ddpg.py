# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from collections import deque
import random
import os
import re # Import regular expressions library

# (The ActorNetwork and CriticNetwork classes remain exactly the same)
class ActorNetwork(tf.keras.Model):
    def __init__(self, M):
        super(ActorNetwork, self).__init__()
        self.M = M
        self.conv1 = layers.Conv2D(32, (1, 3), activation='relu', padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(32, (1, 1), activation='relu', padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(64, activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        self.out = layers.Dense(M, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn3(x)
        x = self.dropout1(x)
        return self.out(x)

class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv1 = layers.Conv2D(32, (1, 3), activation='relu', padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.state_d1 = layers.Dense(64, activation='relu')
        self.action_d1 = layers.Dense(64, activation='relu')
        self.concat = layers.Concatenate()
        self.d2 = layers.Dense(64, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        self.out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

    def call(self, inputs):
        state, action = inputs
        s_out = self.conv1(state)
        s_out = self.bn1(s_out)
        s_out = self.flatten(s_out)
        s_out = self.state_d1(s_out)
        a_out = self.action_d1(action)
        x = self.concat([s_out, a_out])
        x = self.d2(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        return self.out(x)


class DDPG:
    def __init__(self, M, L, N, name, load_weights=False):
        self.name = name
        self.buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.001

        self.actor = ActorNetwork(M)
        self.critic = CriticNetwork()
        self.target_actor = ActorNetwork(M)
        self.target_critic = CriticNetwork()

        dummy_state = tf.random.normal([1, M, L, N])
        dummy_action = self.actor(dummy_state)
        self.critic([dummy_state, dummy_action])
        self.target_actor(dummy_state)
        self.target_critic([dummy_state, dummy_action])
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.mse = tf.keras.losses.MeanSquaredError()

        if load_weights:
            self.load_models()

    def predict(self, state):
        return self.actor(state).numpy()

    def save_transition(self, s, w, r, not_terminal, s_next, action_precise):
        self.buffer.append((s[0], w[0], r, not_terminal, s_next[0], action_precise[0]))

    @tf.function
    def _update_critic(self, states, actions, rewards, next_states, not_terminals):
        target_actions = self.target_actor(next_states, training=True)
        target_q = self.target_critic([next_states, target_actions], training=True)
        y = rewards + self.gamma * not_terminals * target_q
        
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            critic_loss = self.mse(y, q_values)
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        return critic_loss, tf.reduce_max(q_values)

    @tf.function
    def _update_actor(self, states):
        with tf.GradientTape() as tape:
            predicted_actions = self.actor(states, training=True)
            q_values = self.critic([states, predicted_actions], training=True)
            actor_loss = -tf.reduce_mean(q_values)
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        return actor_loss

    def _update_target_networks(self):
        for target_weights, weights in zip(self.target_actor.variables, self.actor.variables):
            target_weights.assign(weights * self.tau + target_weights * (1 - self.tau))
        
        for target_weights, weights in zip(self.target_critic.variables, self.critic.variables):
            target_weights.assign(weights * self.tau + target_weights * (1 - self.tau))

    def train(self, method, epoch):
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": 0, "q_value": 0, "actor_loss": 0}

        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, not_terminals, next_states, _ = map(np.array, zip(*minibatch))

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)[:, None]
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        not_terminals = tf.convert_to_tensor(not_terminals, dtype=tf.float32)[:, None]
        
        critic_loss, q_value = self._update_critic(states, actions, rewards, next_states, not_terminals)
        actor_loss = self._update_actor(states)
        self._update_target_networks()

        return {"critic_loss": critic_loss.numpy(), "q_value": q_value.numpy(), "actor_loss": actor_loss.numpy()}

    def save_models(self, epoch):
        self.actor.save_weights(f'./saved_models/DDPG/{self.name}_actor_{epoch}.weights.h5')
        self.critic.save_weights(f'./saved_models/DDPG/{self.name}_critic_{epoch}.weights.h5')

    # <<< --- THIS IS THE CORRECTED FUNCTION --- >>>
    def load_models(self):
        """
        Loads the weights from the most recent training epoch.
        """
        model_dir = f'./saved_models/DDPG/'
        try:
            # 1. Find all actor weight files for this specific agent configuration
            actor_files = [f for f in os.listdir(model_dir) if f.startswith(f'{self.name}_actor_') and f.endswith('.weights.h5')]

            if not actor_files:
                print("Could not find any model checkpoints. Starting from scratch.")
                return

            # 2. Extract the epoch numbers from the filenames and find the latest one
            latest_epoch = -1
            for f in actor_files:
                # Use regex to find the number between '_' and '.weights'
                match = re.search(r'_(\d+)\.weights', f)
                if match:
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
            
            if latest_epoch == -1:
                print("Could not parse epoch numbers from model files. Starting from scratch.")
                return

            # 3. Construct the full paths to the latest actor and critic models
            actor_path = os.path.join(model_dir, f'{self.name}_actor_{latest_epoch}.weights.h5')
            critic_path = os.path.join(model_dir, f'{self.name}_critic_{latest_epoch}.weights.h5')

            # 4. Load the weights
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Models for epoch {latest_epoch} loaded successfully.")

        except Exception as e:
            print(f"Could not load models due to an error: {e}. Starting from scratch.")