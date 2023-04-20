# v0.0.1-a
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Define the RGAN model
class RGAN():
    def __init__(self, data_shape, latent_dim, generator_output_activation='tanh', discriminator_output_activation='sigmoid'):
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        self.generator_output_activation = generator_output_activation
        self.discriminator_output_activation = discriminator_output_activation
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        z = Input(shape=(self.latent_dim,))
        data = Input(shape=self.data_shape)
        generated_data = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator([generated_data, data])

        self.rgan = Model(inputs=[z, data], outputs=validity)
        self.rgan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):

        model = tf.keras.Sequential()
        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.data_shape), activation=self.generator_output_activation))
        model.add(Reshape(self.data_shape))

        model.summary()
        plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True)
        return model

    def build_discriminator(self):

        data = Input(shape=self.data_shape)
        generated_data = Input(shape=self.data_shape)
        input_data = Concatenate(axis=0)([generated_data, data])

        model = Dense(512)(input_data)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(256)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation=self.discriminator_output_activation)(model)

        model = Model(inputs=[generated_data, data], outputs=validity)

        model.summary()
        plot_model(model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)
        return model
    
    def train(X_train, batch_size, epochs, n_critic, clip_value, latent_dim, n_samples):

    # define the generator
    generator = define_generator(latent_dim)

    # define the critic
    critic = define_critic()

    # define the composite model
    critic.trainable = False
    composite = define_composite(generator, critic)

    # load real data
    X_real = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # calculate the number of batches per training epoch
    bat_per_epo = int(X_real.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    # manually enumerate epochs
    for i in range(epochs):

        # enumerate batches over the training set
        for j in range(bat_per_epo):

            # get randomly selected 'real' samples
            X_real_batch = X_real[j * batch_size:(j + 1) * batch_size]
            
            # generate 'fake' examples
            X_fake_batch = generate_fake_samples(generator, latent_dim, half_batch)

            # update critic
            for _ in range(n_critic):
                c_loss = critic.train_on_batch(X_real_batch, -np.ones((half_batch, 1)))
                c_loss += critic.train_on_batch(X_fake_batch, np.ones((half_batch, 1)))
                for l in critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, batch_size)

            # create inverted labels for the fake samples
            y_gan = -np.ones((batch_size, 1))

            # update the generator via the critic's error
            g_loss = composite.train_on_batch(X_gan, y_gan)

        # evaluate the model performance every 'epoch'
        if (i+1) % 50 == 0:
            # generate 'n_samples' fake samples
            X_fake = generate_fake_samples(generator, latent_dim, n_samples)
            # visualize the first 'n_samples'
            plt.plot(X_fake[0])
            plt.title(f'Generated Samples (Epoch {(i+1)})')
            plt.show()

    # save the generator model
    generator.save('generator_model.h5')