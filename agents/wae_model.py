# agents/wae_model.py
import tensorflow as tf
from tensorflow.keras import layers


def compute_mmd(z):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss.
    This is the core of the Wasserstein Autoencoder's distribution matching.
    It encourages the encoded distribution `z` to match a standard normal distribution.
    """
    prior_z = tf.random.normal(shape=tf.shape(z))

    # Inverse multiquadratics kernel C(x,y) = C / (C + ||x-y||^2)
    def imq_kernel(x, y, c=1.0):
        # Using tf.reduce_sum for squared Euclidean distance
        x_expanded = tf.expand_dims(x, 1)
        y_expanded = tf.expand_dims(y, 0)
        # Broadcasting to get pairwise squared distances
        dist_sq = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
        return c / (c + dist_sq)

    kernel_zz = tf.reduce_mean(imq_kernel(z, z))
    kernel_prior_prior = tf.reduce_mean(imq_kernel(prior_z, prior_z))
    kernel_z_prior = tf.reduce_mean(imq_kernel(z, prior_z))

    return kernel_zz + kernel_prior_prior - 2 * kernel_z_prior


class Encoder(tf.keras.Model):
    """
    Encodes a flattened high-dimensional state into a low-dimensional latent vector 'z'.
    Follows the simple MLP architecture from the paper.
    """

    def __init__(self, state_dim, latent_dim):
        super(Encoder, self).__init__()
        self.net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(state_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(latent_dim)
        ])

    def call(self, state):
        # The environment state is (M, L, N). We flatten it for the MLP.
        flattened_state = layers.Flatten()(state)
        return self.net(flattened_state)


class Decoder(tf.keras.Model):
    """
    Decodes a latent vector 'z' and an action 'a' to predict the next state.
    Follows the simple MLP architecture from the paper.
    """

    def __init__(self, latent_dim, action_dim, state_dim):
        super(Decoder, self).__init__()
        self.state_dim = state_dim
        self.net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim + action_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(state_dim)  # Output is the flattened next state
        ])

    def call(self, z, action):
        za = tf.concat([z, action], axis=1)
        return self.net(za)


class WAE(tf.keras.Model):
    """
    Wasserstein Autoencoder model.
    Contains the Encoder and Decoder and manages the loss calculation.
    """

    def __init__(self, state_shape, action_dim, latent_dim):
        super(WAE, self).__init__()
        self.M, self.L, self.N = state_shape
        self.state_dim_flat = self.M * self.L * self.N
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(self.state_dim_flat, latent_dim)
        self.decoder = Decoder(latent_dim, action_dim, self.state_dim_flat)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def loss_function(self, state, action, next_state, mmd_lambda=2.0):
        """
        Calculates WAE loss = Reconstruction Loss + λ * MMD Loss
        """
        z = self.encoder(state)
        next_state_pred_flat = self.decoder(z, action)

        # Flatten the true next_state for loss calculation
        next_state_flat = layers.Flatten()(next_state)

        # 1. Reconstruction Loss (MSE)
        reconstruction_loss = tf.reduce_mean(tf.square(next_state_flat - next_state_pred_flat))

        # 2. MMD Discrepancy Loss
        mmd_loss = compute_mmd(z)

        total_loss = reconstruction_loss + mmd_lambda * mmd_loss
        return total_loss, reconstruction_loss, mmd_loss