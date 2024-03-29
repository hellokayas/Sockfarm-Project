import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import load_data, split_data_by_time, build_nx, normalize_dict
from detecters import do_fraudar, do_rev2, do_rsd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random attacks on data")
    parser.add_argument("--alg", action="store", type=str, choices=["fraudar", "rsd", "rev2"])
    parser.add_argument("--data", action="store", type=str, choices=["alpha", "otc", "amazon", "epinions"])

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

    args = parser.parse_args()
    args.ctotal = int(args.budget * args.frac // args.ccost)
    args.rtotal = args.req - args.ctotal
    args.rtotal = args.rtotal if args.rtotal > 0 else 0
    args.rbudget = args.budget - args.ctotal * args.ccost
    # args.req = int(args.budget * (1-args.frac) // args.rcost)
    print(args)

    mp.set_start_method("forkserver")
    pool = mp.Pool(processes=args.jobs)

    if args.alg == "fraudar":
        do_alg = do_fraudar
    elif args.alg == "rsd":
        do_alg = do_rsd
    elif args.alg == "rev2":
        do_alg = do_rev2
    else:
        raise NotImplementedError

    output_path = Path(f"../res/no_attack/{args.alg}-{args.data}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"{output_path} exists! Stop and quit")
        exit(1)

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    created_plans = [f"usock{a}" for a in range(args.ctotal)]
    existed = data_gt_df[data_gt_df["label"] == -1]["id"].tolist()

    df_total_list = split_data_by_time(data_nw_df, n_splits=args.total)
    df_splits = [pd.concat(df_total_list[i:args.total-args.splits+i+1]) for i in range(args.splits)]

    # ! randomly selecting the requests for experiment purpose. Set the random seed to a certain value
    np.random.seed(0)
    targets_plans = [np.random.choice(df["dest"][:500], size=args.req, replace=True) for df in df_splits]
    print(f"split shapes: {[df.shape for df in df_splits]}")

    G_list = [build_nx(df) for df in df_splits]
    # ! get the intial scores
    scores = pool.map(func=do_alg, iterable=G_list, chunksize=1)
    scores = [normalize_dict(score) for score in scores]

    # ! only save the users with ground truth, including the socks as well
    output_users = created_plans + data_gt_df["id"].tolist()
    output_scores = [{u: score[u] for u in score if u in output_users} for score in scores]
    with open(output_path, "wb") as fp:
        pickle.dump(output_scores, fp)
