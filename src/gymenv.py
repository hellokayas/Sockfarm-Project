import gym
import numpy as np


class SockFarmEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, size=100, max_step=1):
        '''
        :param size: size of review product matrix
        :param max_step: max step to terminal
        '''
        super().__init__()
        self.pos = 0
        self.size = size
        self.max_step = max_step

        # spaces for action, states, and requests
        self.action_space = gym.spaces.MultiBinary(self.size)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=[self.size]
        )
        self.request_space = gym.spaces.Box(
            low=0, high=1,
            shape=[self.size, self.max_step]
        )

        self.reset()
        return self

    def reset(self):
        self.pos = 0
        self.obs = self.observation_space.sample().astype(np.float)
        self.req = self.request_space.sample().astype(np.float)
        self.tree = []

        # ! need to be changed with real implementation
        self.obs = (self.obs > 0.2) * 1
        self.req = (self.req > 0.05) * 1

        self.obs = self.obs.astype(np.float)
        self.req = self.req.astype(np.float)
        return self.obs

    def step(self, action):
        sum_prev = np.sum(self.obs)
        # * update the observations

        self.obs += action
        reward = np.sum(self.obs) - sum_prev

        self.obs -= self.req[:, self.pos]

        self.obs = np.clip(self.obs, 0, 1)
        self.pos += 1

        done = self.pos == self.max_step

        info = {"info": "info"}
        return self.obs, reward, done, info

    def render(self, mode="console"):
        ret = f"obs: {self.obs}\n"
        ret += f"req: {self.req}"
        # print(ret)
        return ret

    def close(self):
        pass


# run this script directly for testing purpose, otherwise, import the module
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = SockFarmEnv(max_step=10)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())

    from stable_baselines3 import PPO
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=int(1e5))
