# agents/derl.py
import tensorflow as tf
import numpy as np
import copy
import os

from .wae_model import WAE
from .td3 import TD3


class DERLAgent:
    def __init__(self, M, L, N, name, load_weights=False):
        self.name = name
        self.state_shape = (M, L, N)
        self.action_dim = M
        self.latent_dim = 64  # Hyperparameter from the paper (500), but 64 is a good start

        # --- Component 1: WAE for Embedding ---
        self.wae = WAE(self.state_shape, self.action_dim, self.latent_dim)

        # --- Component 2: TD3 for Policy ---
        # The TD3 agent operates on the latent space
        self.td3 = TD3(self.latent_dim, self.action_dim, name, load_weights)

        # --- Component 3: Meta-Learning (FOML) ---
        self.meta_wae = WAE(self.state_shape, self.action_dim, self.latent_dim)
        self.meta_wae.set_weights(self.wae.get_weights())  # Initialize meta with live weights

        # FOML Hyperparameters from the paper
        self.foml_beta1 = 0.001
        self.foml_alpha2 = 0.0005

        self.model_save_path = f'./saved_models/DERL/'
        if load_weights:
            self.load_models()

    def predict(self, state):
        """Selects an action using the TD3 agent on the EMBEDDED state."""
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        latent_state = self.wae.encoder(state_tensor)
        return self.td3.predict(latent_state)

    def pretrain_wae(self, buffer, epochs=1000, batch_size=64):
        print("\n--- Starting WAE Pre-training ---")
        for epoch in range(epochs):
            s, a, _, _, s_prime, _ = buffer.sample(batch_size)
            s = tf.convert_to_tensor(s, dtype=tf.float32)
            a = tf.convert_to_tensor(a, dtype=tf.float32)
            s_prime = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            with tf.GradientTape() as tape:
                loss, recon, mmd = self.wae.loss_function(s, a, s_prime)

            grads = tape.gradient(loss, self.wae.trainable_variables)
            self.wae.optimizer.apply_gradients(zip(grads, self.wae.trainable_variables))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss: {loss:.4f}, Recon: {recon:.4f}, MMD: {mmd:.4f}")

        # Sync meta-model after pre-training
        self.meta_wae.set_weights(self.wae.get_weights())
        print("--- WAE Pre-training Finished. Meta-parameters are set. ---")

    def train_rl(self, buffer, batch_size=100):
        """Trains the TD3 agent on a batch of EMBEDDED experiences."""
        s, a, r, not_done, s_prime, _ = buffer.sample(batch_size)

        # Encode states using the current encoder
        z = self.wae.encoder(tf.convert_to_tensor(s, dtype=tf.float32))
        z_prime = self.wae.encoder(tf.convert_to_tensor(s_prime, dtype=tf.float32))

        # Convert others to tensors
        a = tf.convert_to_tensor(a, dtype=tf.float32)
        r = tf.convert_to_tensor(r, dtype=tf.float32)[:, None]
        not_done = tf.convert_to_tensor(not_done, dtype=tf.float32)[:, None]

        self.td3.train_step_counter += 1
        critic_loss, actor_loss = self.td3.update(z, a, r, z_prime, not_done)
        return {"critic_loss": critic_loss.numpy(), "actor_loss": actor_loss.numpy()}

    def update_embedding_foml(self, recent_buffer, batch_size=40):
        """Performs a dynamic embedding update using the FOML framework."""
        # --- Step 1: Update 'live' WAE parameters (φ, θ) ---
        s, a, _, _, s_prime, _ = recent_buffer.sample(batch_size)
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        a = tf.convert_to_tensor(a, dtype=tf.float32)
        s_prime = tf.convert_to_tensor(s_prime, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss_new_data, _, _ = self.wae.loss_function(s, a, s_prime)

            # Regularization term towards the meta-model (L2 distance)
            meta_reg_loss = tf.reduce_sum([
                tf.reduce_sum(tf.square(param - meta_param))
                for param, meta_param in zip(self.wae.trainable_variables, self.meta_wae.trainable_variables)
            ])

            total_update_loss = loss_new_data + self.foml_beta1 * meta_reg_loss

        grads = tape.gradient(total_update_loss, self.wae.trainable_variables)
        self.wae.optimizer.apply_gradients(zip(grads, self.wae.trainable_variables))

        # --- Step 2: Slowly update 'meta' WAE parameters (ζ) ---
        new_meta_weights = []
        for param, meta_param in zip(self.wae.get_weights(), self.meta_wae.get_weights()):
            new_weight = (1.0 - self.foml_alpha2) * meta_param + self.foml_alpha2 * param
            new_meta_weights.append(new_weight)
        self.meta_wae.set_weights(new_meta_weights)

    def save_best_models(self):
        """Saves both the TD3 actor and the WAE encoder."""
        self.td3.save_best_models()
        self.wae.encoder.save_weights(os.path.join(self.model_save_path, f'{self.name}_encoder_best.weights.h5'))
        print(f"--- New best DERL (TD3+Encoder) model saved ---")

    def load_models(self):
        """Loads both the TD3 actor and the WAE encoder."""
        self.td3.load_models()
        try:
            self.wae.encoder.load_weights(os.path.join(self.model_save_path, f'{self.name}_encoder_best.weights.h5'))
            # Sync the meta model on load
            self.meta_wae.set_weights(self.wae.get_weights())
            print("WAE encoder loaded successfully.")
        except:
            print("Could not find WAE encoder checkpoint. Starting from scratch.")

    def save_transition(self, s, w, r, not_terminal, s_next, action_precise):
        # This method is for compatibility with the main loop, but the DERLAgent doesn't need its own buffer.
        # The buffer is managed in main.py and passed to the training methods.
        pass