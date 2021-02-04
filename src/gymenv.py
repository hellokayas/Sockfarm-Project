import gym
import networkx as nx
import numpy as np
from copy import deepcopy

from utils import normalize_dict
from mypolicy import SockfarmPolicy


class SockFarmEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self,
                 max_step: int = 1,
                 G: nx.DiGraph = None,
                 my_alg=None,
                 out_users=None,
                 socks=None,
                 prods=None,
                 max_requests: int = 100,
                 ) -> None:
        '''
        :param size: size of review product matrix
        :param max_step: max step to terminal
        '''
        super().__init__()

        # * count the step and end at max_step
        self.max_step = max_step
        self.cur_step = 0

        # * save the graph
        self.init_G = G.copy()
        self.G = G.copy()
        self.init_alg = my_alg(self.init_G)
        self.init_alg.update(max_iter=3)
        self.my_alg = deepcopy(self.init_alg)

        # * the users we care about
        self.out_users = [u for u in out_users if u in G.nodes]
        self.socks = socks
        # * products
        self.prods = prods
        self.max_prod = len(self.prods)

        self.max_requests = max_requests

        # spaces for action, states, and requests
        self.action_shape = np.array([len(self.socks), self.max_requests])
        self.action_space = gym.spaces.MultiBinary(self.action_shape[0] * self.action_shape[1])

        # ! observation: detection_probability (dprob) 0-1

        # dropb: len(self.out_users), req: self.max_requests*len(self.prods), rev = req
        self.user_dim = len(self.out_users)
        self.requests_dim = self.max_requests*len(self.prods)
        self.requests_shape = [self.max_requests, len(self.prods)]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=[self.user_dim + self.requests_dim],
        )

        self.init_dprob = normalize_dict(self.my_alg.get_score())

        self.init_obs = np.zeros(shape=self.observation_space.shape)
        self.init_obs[:self.user_dim] = np.array([self.init_dprob[u] for u in self.out_users]).astype(np.float)

        self.reset()

        return None

    def reset(self) -> np.array:
        self.cur_step = 0
        # reset the graph
        self.G = deepcopy(self.init_G)
        self.my_alg = deepcopy(self.init_alg)

        self.obs = np.copy(self.init_obs)

        # * generate a serie of a requests
        self.num_rquests = np.random.multinomial(self.max_requests, np.ones(self.max_step)/self.max_step, size=1)[0]
        # print(f"num_request: {self.num_rquests}")
        self.requests = [np.random.choice(len(self.prods), nreq, replace=True) for nreq in self.num_rquests]

        # init first step
        self.get_reqs()[np.arange(self.num_rquests[self.cur_step]), self.requests[self.cur_step]] = 1

        return self.obs

    def get_reqs(self):
        return self.obs[self.user_dim:].reshape(self.requests_shape)

    def step(self, action: np.array):
        # * update the observations
        # print(action.shape)
        # print(action)

        action = action.reshape(self.action_shape)

        # action \times request -> account*product (review matrix)
        for u, p in np.array(np.where(action@self.get_reqs() > 0.99)).T:
            # ! add review with max rating, for G and algorithm
            # print(f"add review {self.socks[u]}->{self.prods[p]} r:{1}")
            self.G.add_edge(self.socks[u], self.prods[p], rating=1)
            self.my_alg.add_review(self.socks[u], self.prods[p], rating=1)

        for u, r in np.array(np.where(action > 0.99)).T:
            # ! clear request for actions
            # print(f"req0: {self.get_reqs()}")
            self.get_reqs()[r] = 0
            # print(f"req1: {self.get_reqs()}")

        self.my_alg.update(max_iter=1)
        self.dprob = normalize_dict(self.my_alg.get_score())
        self.nobs = np.array([self.dprob[u] for u in self.out_users]).astype(np.float)

        reward = np.sum(self.obs[:self.user_dim] - self.nobs)

        self.cur_step += 1
        done = self.cur_step >= self.max_step

        # ! update obs
        self.obs[:self.user_dim] = self.nobs
        if not done:
            self.get_reqs()[np.arange(self.num_rquests[self.cur_step]), self.requests[self.cur_step]] = 1

        info = {"info": None}

        return self.obs, reward, done, info

    def render(self, mode="console"):
        pass

    def close(self):
        pass


# * run this script directly for testing purpose, otherwise, import the module
if __name__ == "__main__":
    import pandas as pd
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import DDPG

    from algs import rev2_alg as do_alg

    example_nw_df = pd.DataFrame.from_dict({
        "src": ["u1", "u1", "u1", "u2", "u2", "u3"],
        "dest": ["p1", "p2", "p3", "p1", "p2", "p1"],
        # "rating": [-0.2, 0.9, -0.6, 0.1, 0.7, 1],
        "rating": [1, -1, 1, 1, -1, -1]
    })

    G = nx.from_pandas_edgelist(
        example_nw_df,
        source="src",
        target="dest",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    scores = do_alg(G)
    print(scores)

    env = SockFarmEnv(
        max_step=4,
        G=G,
        my_alg=do_alg,
        out_users=["u3", "u2"],
        socks=["u3", "u2"],
        prods=["p1", "p3"],
        max_requests=3,
    )

    # It will check your custom environment and output additional warnings if needed
    obs = env.reset()
    req = env.get_reqs()
    print(f"obs: {obs}")
    print(f"req: {req}")
    act = env.action_space.sample().reshape(env.action_shape)
    print(f"act: {act}")
    print(f"rev: {act@req}")
    env.step(act)

    env.reset()
    check_env(env)

    # model = DDPG("MlpPolicy", env, verbose=1)
    model = DDPG(SockfarmPolicy, env, verbose=1,
                 policy_kwargs={"user_dim": env.user_dim, "max_requests": env.max_requests, "max_prod": env.max_prod}
                 )
    model.learn(total_timesteps=int(1e2))

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    print(env.action_space.shape)
    print(action.shape)

    # print(env.observation_space.sample())
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())
