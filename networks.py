from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def big(observation_space, nb_actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + observation_space))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  return model


def big_he(observation_space, nb_actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + observation_space))
  model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(nb_actions, activation='linear', kernel_initializer='he_normal'))
  return model


def small(observation_space, nb_actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + observation_space))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  return model

def tiny(observation_space, nb_actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + observation_space))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  return model
