import argparse
import pickle
# from multiprocessing import Pool
from pathlib import Path
# from copy import deepcopy

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

from detecters import do_fraudar, do_rev2, do_rsd
from gymenv import SockFarmEnv
from utils import build_nx, load_data, split_data_by_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sockfarm attacks on data")
    parser.add_argument("--alg", action="store", type=str, choices=["fraudar", "rsd", "rev2"], default="fraudar")
    parser.add_argument("--data", action="store", type=str,
                        choices=["alpha", "otc", "amazon", "epinions"], default="alpha")

    parser.add_argument("--jobs", action="store", type=int, default=None, help="multi process")
    parser.add_argument("--splits", action="store", type=int, default=4)
    parser.add_argument("--total", action="store", type=int, default=10, help="total number of splits")

    # parser.add_argument("--acc", action="store", type=int, default=0, help="placeholder")
    parser.add_argument("--prod", action="store", type=int, default=10)
    parser.add_argument("--frac", action="store", type=float, default=0.2)
    parser.add_argument("--req", action="store", type=int, default=100, help="number of requests")

    parser.add_argument("--budget", action="store", type=float, default=100, help="total budget")
    parser.add_argument("--ccost", action="store", type=float, default=5, help="predefined costs for creating")
    parser.add_argument("--rcost", action="store", type=float, default=2, help="predefined costs for rewiewing")

    parser.add_argument("--check_only", action="store_true")

    args = parser.parse_args()
    args.ctotal = int(args.budget // args.ccost)
    # args.ctotal = int(args.budget * args.frac // args.ccost)
    # args.rtotal = args.req - args.ctotal
    # args.rtotal = args.rtotal if args.rtotal > 0 else 0
    # args.rbudget = args.budget - args.ctotal * args.ccost
    # args.req = int(args.budget * (1-args.frac) // args.rcost)
    print(args)

    np.random.seed(0)
    # pool = Pool(processes=args.jobs)

    output_path = Path(f"../res/sockfarm_attack/{args.alg}-{args.data}/{args.budget}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"{output_path} exists! Stop and quit")
        exit(1)

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    created_frauds = [f"usock{a}" for a in range(args.ctotal)]
    created_dummys = [f"udummy{a}" for a in range(3*args.ctotal)]
    existed_frauds = data_gt_df[data_gt_df["label"] == -1]["id"].tolist()

    df_total_list = split_data_by_time(data_nw_df, n_splits=args.total)
    df_splits = [pd.concat(df_total_list[i:args.total-args.splits+i+1]) for i in range(args.splits)]

    G_list = [build_nx(df) for df in df_splits]
    # targets = [np.random.choice(df["dest"], size=args.prod, replace=False) for df in df_splits]
    targets_plans = [np.random.choice(df["dest"][:20], size=args.req, replace=True) for df in df_splits]

    output_users = created_frauds + data_gt_df["id"].tolist() + created_dummys

    for G in G_list:
        for usock in created_frauds + created_dummys:
            G.add_node(usock)

    print(len(data_gt_df["id"].tolist()), len(output_users))

    if args.alg == "fraudar":
        do_alg = do_fraudar
    elif args.alg == "rsd":
        do_alg = do_rsd
    elif args.alg == "rev2":
        do_alg = do_rev2
    else:
        raise NotImplementedError

    print(do_alg)

    scores = []
    for i in range(len(G_list)):
        print(f"split {i}")

        env = SockFarmEnv(
            max_step=4,
            G=G,
            detecter=do_alg,
            out_users=created_frauds + existed_frauds,
            socks=created_frauds + existed_frauds,
            prods=np.unique(targets_plans[0]),
            max_requests=args.req,
        )

        # vec_envs = SubprocVecEnv([lambda: lambda: deepcopy(env) for i in range(4)])

        if args.check_only:
            print("check env")
            check_env(env)
            exit(0)

        model = DDPG("MlpPolicy", env, verbose=1)
        # model = DDPG("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=int(1e2), log_interval=4)
        model.save(f"../res/sockfarm_attack/{args.alg}-{args.data}/m-{args.budget}-{i}")

        print("saved")

        # del model
        # model = DDPG.load(f"../res/sockfarm_attack/{args.alg}-{args.data}/m-{args.budget}-{i}")
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        score = env.dprob
        scores += [score]

    output_scores = [{u: score[u] for u in score if u in output_users} for score in scores]
    with open(output_path, "wb") as fp:
        pickle.dump(output_scores, fp)
