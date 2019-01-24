from keras.callbacks import TensorBoard as KerasTensorBoard
from rl.callbacks import Callback
from array2gif import write_gif

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

class TensorBoard(KerasTensorBoard):

  def on_train_begin(self, logs):
    self.metrics_names = self.model.metrics_names
    self.image_summaries = []
    self.best_reward = 0

    for layer in self.model.layers:
      for weight in layer.weights:
        name = weight.name.replace(':','_')
        w_img = tf.squeeze(weight)
        shape = K.int_shape(w_img)
        if len(shape) == 2:  # dense layer kernel case
          if shape[0] > shape[1]:
              w_img = tf.transpose(w_img)
              shape = K.int_shape(w_img)
          w_img = tf.reshape(w_img, [1, shape[0], shape[1], 1])
          self.image_summaries.append(tf.summary.image(name, w_img, max_outputs=1))
    
    self.obs_img = tf.placeholder(tf.float32)
    obs_reshaped = tf.reshape(self.obs_img, [1, 8, 16, 1])
    self.image_summaries.append(tf.summary.image('observation', obs_reshaped, max_outputs=1))

  
  def on_episode_begin(self, episode, logs={}):
    """Called at beginning of each episode"""
    self.metrics = []
    self._observations = []

  def on_episode_end(self, episode, logs={}):
    global_step = logs['nb_steps']
    """Called at end of each episode"""
    metric_means = np.nanmean(self.metrics, axis=0)
    summaries = [a for a in zip(self.metrics_names, metric_means)] + [a for a in logs.items()]
    if logs['episode_reward'] > self.best_reward:
      self.best_reward = logs['episode_reward']
      summaries.append(['best_reward', self.best_reward])
    
    for name, value in summaries:
      summary = tf.Summary()
      summary_value = summary.value.add()
      if isinstance(value, np.ndarray):
        summary_value.simple_value = value.item()
      else:
        summary_value.simple_value = value
      summary_value.tag = name
      self.writer.add_summary(summary, global_step)
    if episode % 1000 == 0:
      self._write_images(global_step)

  def on_step_end(self, step, logs={}):
    """Called at end of each step"""
    self.metrics.append(logs['metrics'])
    self._observations.append(logs['observation'])
    pass

  def _write_images(self, episode):
    obs = self._observations[np.random.choice(len(self._observations))]
    summaries = K.get_session().run(self.image_summaries, feed_dict={ self.obs_img: obs })
    [self.writer.add_summary(summary, episode) for summary in summaries]




class Visualizer(Callback):
  def __init__(self, name):
    self.name = name
    self.screens = []
    self.best_screens = []
    self.best_score = -1

  def on_episode_begin(self, episode, logs):
    self.screens = []
  
  def on_step_end(self, step, logs):
    if (step % 4 == 0):
      frame = self.env.render(mode='rgb_array')
      frame = np.transpose(frame, (1,0,2))
      self.screens.append(frame)
  
  def on_episode_end(self, episode, logs):
    if logs['episode_reward'] > self.best_score:
      self.best_score = logs['episode_reward']
      self.best_screens = self.screens

  def write(self):
    write_gif(self.best_screens, 'recordings/{}.gif'.format(self.name), fps=30)


class EpochTest(Callback):
  def __init__(self, model):
    self.model = model
    self.epoch = None
    self.epochs = {}

  def set_epoch(self, epoch):
    self.epoch = epoch
  
  def on_episode_end(self, episode, logs):
    if not self.epoch in self.epochs:
      self.epochs[self.epoch] = []
    self.epochs[self.epoch].append(logs['episode_reward'])
  