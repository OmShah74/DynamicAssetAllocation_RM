import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
import os

class ActorCriticNetwork(tf.keras.Model):
    """
    More advanced Actor-Critic network with deeper layers.
    """
    def __init__(self, M):
        super(ActorCriticNetwork, self).__init__()
        # Shared CNN layers
        self.conv1 = layers.Conv2D(64, (1, 3), activation='relu', padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, (1, 1), activation='relu', padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.shared_dense = layers.Dense(128, activation='relu')

        # Actor head
        self.actor_dense = layers.Dense(128, activation='relu')
        self.actor_mu = layers.Dense(M, activation='tanh')
        self.actor_sigma = layers.Dense(M, activation='softplus')

        # Critic head
        self.critic_dense = layers.Dense(128, activation='relu')
        self.critic_value = layers.Dense(1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x = self.shared_dense(x)
        
        # Actor pathway
        actor_x = self.actor_dense(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x) + 1e-5 # Add epsilon for stability

        # Critic pathway
        critic_x = self.critic_dense(x)
        value = self.critic_value(critic_x)
        
        return mu, sigma, value

class PPO:
    def __init__(self, M, L, N, name, load_weights=False):
        self.name = name
        self.gamma = 0.99
        self.lam = 0.95 # Lambda for GAE
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01 # Coefficient for entropy bonus
        self.update_steps = 10
        self.model_save_path = f'./saved_models/PPO/'
        
        self.model = ActorCriticNetwork(M)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Slower learning rate for stability
        self.mse = tf.keras.losses.MeanSquaredError()

        # Build model
        self.model(tf.random.normal([1, M, L, N]))

        if load_weights:
            self.load_models()
            
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}

    def predict(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        raw_action = dist.sample()
        action_weights = tf.nn.softmax(raw_action)
        return raw_action.numpy(), action_weights.numpy()

    def save_transition(self, s, raw_action, r, not_terminal, s_next):
        mu, sigma, value = self.model(s)
        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(raw_action)

        self.buffer['states'].append(s[0])
        self.buffer['actions'].append(raw_action[0])
        self.buffer['rewards'].append(r)
        self.buffer['values'].append(value.numpy()[0, 0])
        self.buffer['log_probs'].append(log_prob.numpy()[0])
    
    def _calculate_gae(self, rewards, values, not_terminals, last_state_value):
        """
        Calculates the Generalized Advantage Estimation (GAE).
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        
        # Append the value of the final state to the values list for calculation
        values_with_last = np.append(values, last_state_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_last[t+1] * not_terminals[t] - values_with_last[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * not_terminals[t] * last_advantage
        
        # The discounted rewards are just the advantages plus the values
        discounted_rewards = advantages + values
        return advantages, discounted_rewards

    @tf.function
    def _update_step(self, states, actions, discounted_rewards, old_log_probs, advantages):
        with tf.GradientTape() as tape:
            mu, sigma, values = self.model(states, training=True)
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            
            new_log_probs = dist.log_prob(actions)
            
            # Reshape advantages to match the shape of log_probs for element-wise multiplication
            advantages_reshaped = tf.reshape(advantages, (-1, 1))

            # Actor Loss (Clipped Surrogate Objective)
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages_reshaped
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_reshaped
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic Loss (Value Function Loss)
            critic_loss = self.mse(discounted_rewards, values)

            # Entropy Bonus for Exploration
            entropy = tf.reduce_mean(dist.entropy())
            
            # Total Loss
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return critic_loss

    def train(self, method, epoch):
        if not self.buffer['rewards']:
            return {"critic_loss": 0}
        
        last_state = self.buffer['states'][-1][np.newaxis, ...]
        _, _, last_state_value = self.model(last_state)
        
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        not_terminals = np.ones_like(rewards) # Assume all steps are non-terminal until the end
        
        advantages, discounted_rewards = self._calculate_gae(rewards, values, not_terminals, last_state_value.numpy()[0, 0])
        
        states = tf.convert_to_tensor(np.array(self.buffer['states']), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.buffer['actions']), dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(np.array(self.buffer['log_probs']), dtype=tf.float32)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards[:, np.newaxis], dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        critic_loss = 0
        for _ in range(self.update_steps):
            loss = self._update_step(states, actions, discounted_rewards, old_log_probs, advantages)
            critic_loss += loss.numpy()

        self.reset_buffer()
        return {"critic_loss": critic_loss / self.update_steps}

    # <<< START: MODIFIED SECTION >>>
    # This function is now specifically for saving the BEST model
    def save_best_models(self):
        """Saves the weights of the current model as the best model."""
        model_path = os.path.join(self.model_save_path, f'{self.name}_model_best.weights.h5')
        self.model.save_weights(model_path)
        print(f"--- New best model saved ---")

    # This function is now simplified to only load the BEST model
    def load_models(self):
        """Loads the best available model weights."""
        try:
            model_path = os.path.join(self.model_save_path, f'{self.name}_model_best.weights.h5')
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
                print(f"Best PPO model loaded successfully from {model_path}.")
            else:
                print("Could not find a best model checkpoint. Starting from scratch.")
        except Exception as e:
            print(f"Could not load PPO model. Error: {e}. Starting from scratch.")
    # <<< END: MODIFIED SECTION >>>