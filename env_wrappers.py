import gym

class ScaleWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

    self.observation_space.low = [0 for _ in range(128)]
    self.observation_space.high = [1 for _ in range(128)]

  def observation(self, observation):
    return observation / 255.0

# Adapted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# Instead of resuming an ongoing episode on reset, always hard reset the environment


class EpisodicLifeEnv(gym.Wrapper):
  def __init__(self, env):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    gym.Wrapper.__init__(self, env)
    self.lives = 0
    self.was_real_done = True

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.was_real_done = done
    # check current lives, make loss of life terminal,
    # then update lives to handle bonus lives
    lives = self.env.unwrapped.ale.lives()
    if lives < self.lives and lives > 0:
      # for Qbert sometimes we stay in lives == 0 condition for a few frames
      # so it's important to keep lives > 0, so that we only reset once
      # the environment advertises done.
      done = True
    self.lives = lives
    return obs, reward, done, info

  def reset(self, **kwargs):
    """Reset only when lives are exhausted.
    This way all states are still reachable even though lives are episodic,
    and the learner need not know about any of this behind-the-scenes.
    """
    if self.was_real_done:
      obs = self.env.reset(**kwargs)
    else:
      # no-op step to advance from terminal/lost life state
      obs, _, _, _ = self.env.step(0)
    self.lives = self.env.unwrapped.ale.lives()
    return obs
