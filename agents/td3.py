# agents/td3.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(action_dim, activation='softmax') # Softmax for portfolio weights

    def call(self, state_embedding):
        x = self.d1(state_embedding)
        x = self.d2(x)
        return self.out(x)

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(1)

    def call(self, inputs):
        state_embedding, action = inputs
        x = tf.concat([state_embedding, action], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

class TD3:
    def __init__(self, latent_dim, action_dim, name, load_weights=False):
        self.name = name
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_delay = 2
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.model_save_path = f'./saved_models/DERL/'

        # Create actor and two critics
        self.actor = Actor(action_dim)
        self.critic1 = Critic()
        self.critic2 = Critic()

        # Create target networks
        self.target_actor = Actor(action_dim)
        self.target_critic1 = Critic()
        self.target_critic2 = Critic()

        # Build and copy weights
        dummy_embedding = tf.random.normal([1, latent_dim])
        self.actor(dummy_embedding)
        self.critic1([dummy_embedding, self.actor(dummy_embedding)])
        self.critic2([dummy_embedding, self.actor(dummy_embedding)])
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.train_step_counter = 0

    def predict(self, state_embedding):
        return self.actor(state_embedding).numpy()

    @tf.function
    def update(self, z, a, r, z_prime, not_done):
        # --- Train Critic ---
        with tf.GradientTape(persistent=True) as tape:
            # Target policy smoothing
            noise = tf.clip_by_value(tf.random.normal(a.shape) * self.policy_noise, -self.noise_clip, self.noise_clip)
            target_a = tf.clip_by_value(self.target_actor(z_prime) + noise, 0, 1) # Clip to valid action range
            target_a = target_a / tf.reduce_sum(target_a, axis=1, keepdims=True) # Re-normalize

            # Double Q-learning: take the minimum of the two target critics
            target_q1 = self.target_critic1([z_prime, target_a])
            target_q2 = self.target_critic2([z_prime, target_a])
            target_q = r + not_done * self.gamma * tf.minimum(target_q1, target_q2)

            current_q1 = self.critic1([z, a])
            current_q2 = self.critic2([z, a])
            
            critic_loss = tf.reduce_mean(tf.square(current_q1 - target_q)) + \
                          tf.reduce_mean(tf.square(current_q2 - target_q))
        
        critic_grad = tape.gradient(critic_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic1.trainable_variables + self.critic2.trainable_variables))
        
        # --- Delayed Actor Training ---
        actor_loss = tf.constant(0.0)
        if self.train_step_counter % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(self.critic1([z, self.actor(z)]))
            
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            
            # Soft update target networks
            self._update_target_networks()

        return critic_loss, actor_loss

    def _update_target_networks(self):
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(source * self.tau + target * (1 - self.tau))
        for target, source in zip(self.target_critic1.variables, self.critic1.variables):
            target.assign(source * self.tau + target * (1 - self.tau))
        for target, source in zip(self.target_critic2.variables, self.critic2.variables):
            target.assign(source * self.tau + target * (1 - self.tau))

    def save_best_models(self):
        self.actor.save_weights(os.path.join(self.model_save_path, f'{self.name}_actor_best.weights.h5'))

    def load_models(self):
        try:
            self.actor.load_weights(os.path.join(self.model_save_path, f'{self.name}_actor_best.weights.h5'))
            # Re-sync target networks after loading
            self.target_actor.set_weights(self.actor.get_weights())
            print(f"DERL models loaded successfully.")
        except:
            print("Could not find DERL model checkpoint. Starting from scratch.")