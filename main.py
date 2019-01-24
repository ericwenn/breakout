import os
import datetime
import sys
import numpy as np
import gym

from keras.optimizers import Adam

from callbacks import Visualizer, TensorBoard, EpochTest
from env_wrappers import ScaleWrapper, EpisodicLifeEnv
from networks import big_he

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

MODE = 'train'
if len(sys.argv) == 2 and sys.argv[1] == 'vis':
  MODE = 'vis'
if len(sys.argv) == 2 and sys.argv[1] == 'test':
  MODE = 'test'

ENV_NAME = 'Breakout-ram-v0'
MODEL_NAME = 'lr_5e-5_hard_25e4'
LEARNING_RATE = 5e-5
TARGET_MODEL_UPDATE = 2.5e4
DOUBLE_DQN = False
MEMORY_SIZE = 1000000
NETWORK=big_he
EPS_MAX = 1.0
EPS_MIN = .1
EPS_DECAY_STEPS = 1e6
EPS_TEST = 0.05
GAMMA = .95
TRAINING_STEPS = 5e6

TB_DIR = '{}/tf-logs/{}'.format(os.getenv("HOME"), MODEL_NAME)
MODEL_DIR = 'models'
WEIGHT_CHECKPOINT_FILE = MODEL_DIR + '/' + MODEL_NAME + '_{step}.h5f'
WEIGHT_FINAL_FILE = MODEL_DIR + '/' +MODEL_NAME + '.h5f'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
print(env.unwrapped.get_action_meanings())
if MODE == 'train':
  env = ScaleWrapper(EpisodicLifeEnv(env))
else:
  env = ScaleWrapper(env)


nb_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Seed
np.random.seed(123)
env.seed(123)

# Next, we build a very simple model.
model = NETWORK(obs_shape, nb_actions)
print(model.summary())


memory = SequentialMemory(
  limit=MEMORY_SIZE, 
  window_length=1
)

policy = LinearAnnealedPolicy(
  EpsGreedyQPolicy(),
  attr='eps', 
  value_max=EPS_MAX,
  value_min=EPS_MIN,
  value_test=EPS_TEST,
  nb_steps=EPS_DECAY_STEPS
)

dqn = DQNAgent(
  model=model, 
  gamma=GAMMA,
  nb_actions=nb_actions, 
  memory=memory, 
  nb_steps_warmup=1000,
  target_model_update=TARGET_MODEL_UPDATE,
  policy=policy,
  test_policy=policy,
  enable_double_dqn=DOUBLE_DQN
)

dqn.compile(
  optimizer=Adam(lr=LEARNING_RATE), 
  metrics=['mae']
)

if __name__ == "__main__":
  if MODE == 'train':
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    callbacks = [
      ModelIntervalCheckpoint(WEIGHT_CHECKPOINT_FILE, interval=50000),
      TensorBoard(log_dir=TB_DIR)
    ]

    dqn.fit(
      env,
      callbacks=callbacks,
      nb_steps=TRAINING_STEPS,
      log_interval=10000
    )

    # After training is done, we save the final weights.
    dqn.save_weights(WEIGHT_FINAL_FILE, overwrite=True)



  elif MODE == 'vis':
    dqn.load_weights(WEIGHT_FINAL_FILE)
    vis = Visualizer(name=MODEL_NAME)
    dqn.test(env, nb_episodes=100, visualize=False, callbacks=[vis])
    vis.write()

  elif MODE == 'test':
    from glob import glob
    import re
    patt = './{}/{}_*.h5f'.format(MODEL_DIR, MODEL_NAME)
    print(patt)
    # Every filename should have a contain a number denoting the step
    ws = sorted(glob(patt), key=lambda fn: int(re.findall(r"(\d+)\.h5f", fn)[0]))
    epochtest = EpochTest(MODEL_NAME)
    vis = Visualizer(name=MODEL_NAME)
    for w in ws:
      epoch = int(re.findall(r"(\d+)\.h5f", w)[0])
      print('Epoch: {}'.format(epoch))
      epochtest.set_epoch(epoch)
      dqn.load_weights(w)
      dqn.test(env, nb_episodes=10, visualize=False, callbacks=[epochtest, vis])
    
    vis.write()
    import json

    with open('evaluations/{}.json'.format(MODEL_NAME), 'w') as outfile:
        json.dump(epochtest.epochs, outfile)


