import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_data, split_data_by_time, build_nx
from detectoers import do_feagle, do_fraudar, do_rev2, do_rsd


def no_attack(df, socks, n_prod=10, n_req=100):
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

    parser.add_argument("--jobs", action="store", type=int, default=None)
    parser.add_argument("--splits", action="store", type=int, default=5)
    parser.add_argument("--total", action="store", type=int, default=10)
    parser.add_argument("--req", action="store", type=int, default=100)
    parser.add_argument("--acc", action="store", type=int, default=10)
    parser.add_argument("--prod", action="store", type=int, default=10)

    args = parser.parse_args()
    print(args)
    np.random.seed(0)
    pool = Pool(processes=args.jobs)

    output_path = Path(f"../res/no_attack/{args.alg}-{args.data}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"{output_path} exists! Stop and quit")
        exit(1)

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    longest = data_nw_df["src"].map(lambda x: len(x)).max()
    print(longest)
    socks = [f"usock{a}" for a in range(args.acc)]

    df_total_list = split_data_by_time(data_nw_df, n_splits=args.total)
    df_splits = [pd.concat(df_total_list[i:args.total-args.splits+i+1]) for i in range(args.splits)]
    df_attack = df_splits
    G_list = [build_nx(df) for df in df_attack]
    for G in G_list:
        G.add_nodes_from(socks)

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
