# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, M):
        super(ActorCriticNetwork, self).__init__()
        # Shared CNN layers
        self.conv1 = layers.Conv2D(32, (1, 3), activation='relu', padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        
        # Actor head
        self.actor_dense = layers.Dense(64, activation='relu')
        self.actor_mu = layers.Dense(M, activation='tanh') # Mean of the action distribution
        self.actor_sigma = layers.Dense(M, activation='softplus') # Std dev of the action distribution

        # Critic head
        self.critic_dense = layers.Dense(64, activation='relu')
        self.critic_value = layers.Dense(1) # Value estimate

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.flatten(x)
        
        # Actor pathway
        actor_x = self.actor_dense(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)

        # Critic pathway
        critic_x = self.critic_dense(x)
        value = self.critic_value(critic_x)
        
        return mu, sigma, value

class PPO:
    def __init__(self, M, L, N, name, load_weights=False):
        self.name = name
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.update_steps = 10
        
        self.model = ActorCriticNetwork(M)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.mse = tf.keras.losses.MeanSquaredError()

        # Build model
        self.model(tf.random.normal([1, M, L, N]))

        if load_weights:
            self.load_models()
            
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}

    def predict(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfp.distributions.Normal(loc=mu, scale=sigma + 1e-5)
        action = dist.sample()
        
        # Convert action to portfolio weights (must sum to 1)
        action_weights = tf.nn.softmax(action)
        return action_weights.numpy()

    def save_transition(self, s, w, r, not_terminal, s_next, action_precise):
        # For PPO, we store transitions for the whole episode
        mu, sigma, _ = self.model(s)
        dist = tfp.distributions.Normal(loc=mu, scale=sigma + 1e-5)
        log_prob = dist.log_prob(tf.math.log(w / (1-w + 1e-8))) # Inverse softmax might be unstable, using action directly is better if available

        self.buffer['states'].append(s[0])
        self.buffer['actions'].append(w[0])
        self.buffer['rewards'].append(r)
        self.buffer['log_probs'].append(log_prob.numpy()[0])
    
    def _calculate_discounted_rewards(self, rewards, last_state_value):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = last_state_value
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.gamma * running_add
            discounted_rewards[t] = running_add
        return discounted_rewards

    @tf.function
    def _update_step(self, states, actions, discounted_rewards, old_log_probs):
        with tf.GradientTape() as tape:
            mu, sigma, values = self.model(states, training=True)
            dist = tfp.distributions.Normal(loc=mu, scale=sigma + 1e-5)
            
            # Inverse softmax approx.
            action_logits = tf.math.log(actions / (1-actions + 1e-8)) 
            new_log_probs = dist.log_prob(action_logits)

            advantages = discounted_rewards - values
            
            # Actor Loss (Clipped Surrogate Objective)
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic Loss
            critic_loss = self.mse(discounted_rewards, values)

            # Total Loss (can add entropy bonus for exploration)
            total_loss = actor_loss + 0.5 * critic_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return critic_loss

    def train(self, method, epoch):
        if not self.buffer['rewards']:
            return {"critic_loss": 0}
        
        # Get value of the last state to bootstrap rewards
        last_state = self.buffer['states'][-1][np.newaxis, ...]
        _, _, last_state_value = self.model(last_state)
        
        rewards = self.buffer['rewards']
        discounted_rewards = self._calculate_discounted_rewards(rewards, last_state_value.numpy()[0, 0])
        
        # Prepare tensors for training
        states = tf.convert_to_tensor(np.array(self.buffer['states']), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.buffer['actions']), dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(np.array(self.buffer['log_probs']), dtype=tf.float32)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards[:, np.newaxis], dtype=tf.float32)
        
        critic_loss = 0
        for _ in range(self.update_steps):
            loss = self._update_step(states, actions, discounted_rewards, old_log_probs)
            critic_loss += loss.numpy()

        self.reset_buffer()
        return {"critic_loss": critic_loss / self.update_steps}

    def save_models(self, epoch):
        self.model.save_weights(f'./saved_models/PPO/{self.name}_model_{epoch}.weights.h5')

    def load_models(self):
        try:
            self.model.load_weights(f'./saved_models/PPO/{self.name}_model.weights.h5')
            print("Model loaded successfully.")
        except:
            print("Could not load model. Starting from scratch.")