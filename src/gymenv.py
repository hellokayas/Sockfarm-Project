import gym
import networkx as nx
import numpy as np


class SockFarmEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self,
                 act_size: int = 20,
                 max_step: int = 1,
                 G: nx.DiGraph = None,
                 detecter=None,
                 out_users=None,
                 socks=None,
                 prods=None,
                 ) -> None:
        '''
        :param size: size of review product matrix
        :param max_step: max step to terminal
        '''
        super().__init__()
        self.pos = 0
        self.max_step = max_step
        self.init_G = G.copy()
        self.G = G.copy()
        self.out_users = [u for u in out_users if u in G.nodes]
        self.detecter = detecter
        self.socks = socks
        self.prods = prods

        # spaces for action, states, and requests
        self.action_space = gym.spaces.MultiBinary(len(self.socks)*len(self.prods))
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=[len(self.out_users)]
        )

        # print(self.action_space.n)

        self.init_dprob = self.detecter(self.G)

        self.init_obs = np.array([self.init_dprob[u] for u in self.out_users]).astype(np.float)

        self.reset()
        return None

    def reset(self) -> np.array:
        self.pos = 0
        self.G = self.init_G.copy()

        self.obs = np.copy(self.init_obs)
        return self.obs

    def step(self, action: np.array):
        # * update the observations
        # print(action.shape)
        # print(action)
        action = action.reshape([len(self.socks), len(self.prods)])
        for u, p in np.array(np.where(action == 1)).T:
            # ! add review with max rating
            self.G.add_edge(self.socks[u], self.prods[p], rating=1)

        self.dprob = self.detecter(self.G)
        self.nobs = np.array([self.dprob[u] for u in self.out_users]).astype(np.float)

        reward = np.sum(self.nobs - self.obs)

        done = True

        info = {"info": "info"}
        return self.nobs, reward, done, info

    def render(self, mode="console"):
        pass

    def close(self):
        pass


# ! outdated
# run this script directly for testing purpose, otherwise, import the module
# if __name__ == "__main__":
#     from stable_baselines3.common.env_checker import check_env

#     env = SockFarmEnv(max_step=10)
#     # It will check your custom environment and output additional warnings if needed
#     check_env(env)
#     # env.step(env.action_space.sample())
#     # env.step(env.action_space.sample())

#     from stable_baselines3 import PPO
#     model = PPO('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=int(1e2))
