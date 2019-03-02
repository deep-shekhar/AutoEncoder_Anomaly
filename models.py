from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D

def autoencoder():
    input_shape=(40000,)
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=input_shape))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(40000, activation='sigmoid'))
    return model

def deep_autoencoder():
    input_shape=(40000,)
    model = Sequential()
    model.add(Dense(12800, activation='relu', input_shape=input_shape))
    model.add(Dense(6400, activation='relu'))
    model.add(Dense(12800, activation='relu'))
    model.add(Dense(40000, activation='sigmoid'))
    return model

def convolutional_autoencoder():

    input_shape=(200,200,1)
    n_channels = input_shape[-1]
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(16, (5,5), activation='relu', padding='same'))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(8, (5,5), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(16, (5,5), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(Conv2D(n_channels, (5,5), activation='sigmoid', padding='same'))
    return model

def load_model(name):
    if name=='autoencoder':
        return autoencoder()
    elif name=='deep_autoencoder':
        return deep_autoencoder()
    elif name=='convolutional_autoencoder':
        return convolutional_autoencoder()
    else:
        raise ValueError('Unknown model name %s was given' % name)
