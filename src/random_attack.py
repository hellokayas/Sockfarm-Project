import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import load_data, split_data_by_time, build_nx
from detecters import do_feagle, do_fraudar, do_rev2, do_rsd


def naive_attack(df, socks, n_prod=10, n_req=100):
    targets = np.random.choice(df["dest"], size=n_prod, replace=False)
    rdf = pd.DataFrame(
        [
            {
                "src": s,
                "dest": t,
                "rating": df["rating"].max(),
                "timestamp": df["timestamp"].max(),
            }
            for s, t in zip(np.random.choice(socks, size=n_req), np.random.choice(targets, size=n_req))
        ]
    )
    return rdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random attacks on data")
    parser.add_argument("--alg", action="store", type=str, choices=["fraudar", "feagle", "rsd", "rev2"])
    parser.add_argument("--data", action="store", type=str, choices=["alpha", "otc", "amazon", "epinions"])

    parser.add_argument("--jobs", action="store", type=int, default=None, help="multi process")
    parser.add_argument("--splits", action="store", type=int, default=4)
    parser.add_argument("--total", action="store", type=int, default=10, help="total number of splits")

    parser.add_argument("--acc", action="store", type=int, default=0, help="placeholder")
    parser.add_argument("--prod", action="store", type=int, default=10)
    parser.add_argument("--frac", action="store", type=float, default=0.2)
    parser.add_argument("--req", action="store", type=int, default=0, help="placeholder")

    parser.add_argument("--budget", action="store", type=float, default=100, help="total budget")
    parser.add_argument("--ccost", action="store", type=float, default=5, help="predefined")
    parser.add_argument("--rcost", action="store", type=float, default=1, help="predefined")

    args = parser.parse_args()
    args.acc = int(args.budget * args.frac // args.ccost)
    args.req = int(args.budget * (1-args.frac) // args.rcost)
    print(args)

    np.random.seed(0)
    mp.set_start_method("forkserver")
    pool = mp.Pool(processes=args.jobs)

    output_path = Path(f"../res/naive_attack/{args.alg}-{args.data}/{args.budget}-{args.frac}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"{output_path} exists! Stop and quit")
        exit(1)

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    socks = [f"usock{a}" for a in range(args.acc)]

    df_total_list = split_data_by_time(data_nw_df, n_splits=args.total)
    df_splits = [pd.concat(df_total_list[i:args.total-args.splits+i+1]) for i in range(args.splits)]
    df_attack = [pd.concat([df, naive_attack(df, socks=socks, n_prod=args.prod, n_req=args.req)])
                 for df in df_splits]
    G_list = [build_nx(df) for df in df_attack]

    # ! parallel run the chunks
    if args.alg == "fraudar":
        scores = pool.map(func=do_fraudar, iterable=G_list, chunksize=1)
    elif args.alg == "feagle":
        scores = pool.map(func=do_feagle, iterable=G_list, chunksize=1)
    elif args.alg == "rsd":
        scores = pool.map(func=do_rsd, iterable=G_list, chunksize=1)
    elif args.alg == "rev2":
        scores = pool.map(func=do_rev2, iterable=G_list, chunksize=1)

    # ! only save the users with ground truth, including the socks as well
    output_users = socks + data_gt_df["id"].tolist()
    output_scores = [{u: score[u] for u in score if u in output_users} for score in scores]
    with open(output_path, "wb") as fp:
        pickle.dump(output_scores, fp)
