# agents/embedding_model.py
import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    """Encodes a high-dimensional state into a latent distribution."""
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="enc_conv1")
        self.lstm = layers.LSTM(64, name="enc_lstm")
        self.dense_mean = layers.Dense(latent_dim, name="enc_mean")
        self.dense_log_var = layers.Dense(latent_dim, name="enc_log_var")

    def call(self, inputs):
        # We process each asset's time series and concatenate the results
        all_asset_features = []
        num_assets = inputs.shape[1]
        for i in range(num_assets):
            asset_series = inputs[:, i, :, :]
            x = self.conv1(asset_series)
            x = self.lstm(x)
            all_asset_features.append(x)
        
        combined = tf.concat(all_asset_features, axis=1)
        mean = self.dense_mean(combined)
        log_var = self.dense_log_var(combined)
        return mean, log_var

class Decoder(tf.keras.Model):
    """Decodes a latent vector and an action to predict the next state."""
    def __init__(self, original_shape):
        super(Decoder, self).__init__()
        self.M, self.L, self.N = original_shape
        self.dense1 = layers.Dense(256, activation='relu', name="dec_dense1")
        self.repeat = layers.RepeatVector(self.L * self.M)
        self.lstm_decoder = layers.LSTM(self.N, return_sequences=True, name="dec_lstm")
        self.reshape = layers.Reshape((self.M, self.L, self.N))

    def call(self, inputs):
        latent_vector, action = inputs
        # Combine the latent vector and the action
        combined = tf.concat([latent_vector, action], axis=1)
        x = self.dense1(combined)
        x = self.repeat(x)
        x = self.lstm_decoder(x)
        # Reshape to match the original state shape
        reconstructed_s_prime = self.reshape(x)
        return reconstructed_s_prime

class VariationalAutoencoder(tf.keras.Model):
    """Combines the Encoder and Decoder into a single VAE model."""
    def __init__(self, original_shape, latent_dim=32):
        super(VariationalAutoencoder, self).__init__()
        self.original_shape = original_shape
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_shape)
        self.vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean

    @tf.function
    def train_step(self, data):
        s, a, s_prime = data
        with tf.GradientTape() as tape:
            mean, log_var = self.encoder(s)
            z = self.reparameterize(mean, log_var)
            s_prime_pred = self.decoder([z, a])
            
            # Reconstruction loss: how well we predict the next state
            recon_loss = tf.reduce_mean(tf.square(s_prime - s_prime_pred))
            
            # KL divergence loss: forces latent space to be a smooth gaussian
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
            
            total_loss = recon_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return total_loss