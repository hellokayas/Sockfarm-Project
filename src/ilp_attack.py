import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import load_data, split_data_by_time, build_nx, normalize_dict
from detecters import do_fraudar, do_rev2, do_rsd, do_fbox, do_sg

import cvxpy


def do_attack(G, targets, creates_fraud, exists, creates_dummy):
    print(len(G.nodes))
    for usock in creates_fraud + creates_dummy:
        G.add_node(usock)
    print(len(G.nodes))
    plans = sum([[u]*exists[u] for u in exists], []) + creates_fraud
    plans = np.random.permutation(plans)
    # print(f"{len(plans)}: {plans}")
    for u, p in zip(plans, targets):
        G.add_edge(u, p, rating=1)
    return G


def ILPsolve(prices, req, rbudget) -> dict:
    ulist = list(prices.keys())
    x = cvxpy.Variable(shape=len(ulist), name="x", integer=True)
    y = np.array([prices[u] for u in ulist])
    problem = cvxpy.Problem(objective=cvxpy.Maximize(x@y), constraints=[cvxpy.sum(x) <= req, x@y <= rbudget, x >= 0])
    # problem.solve(solver=cvxpy.GUROBI)
    # problem.solve(solver=cvxpy.GLPK_MI)
    problem.solve()
    print(f"STATUS: {problem.status} VALUE: {problem.value}")
    return dict(zip(ulist, x.value.astype(int)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random attacks on data")
    parser.add_argument("--alg", action="store", type=str,
                        choices=["fraudar", "rsd", "rev2", "fbox", "sg"], default="fraudar")
    parser.add_argument("--data", action="store", type=str,
                        choices=["alpha", "otc", "amazon", "epinions"], default="alpha")

    parser.add_argument("--jobs", action="store", type=int, default=None, help="multi process")
    parser.add_argument("--splits", action="store", type=int, default=4)
    parser.add_argument("--total", action="store", type=int, default=10, help="total number of splits")

    # parser.add_argument("--acc", action="store", type=int, default=0, help="placeholder")
    parser.add_argument("--prod", action="store", type=int, default=10)
    parser.add_argument("--frac", action="store", type=float, default=0.2)
    parser.add_argument("--req", action="store", type=int, default=100, help="number of requests")

    parser.add_argument("--outdir", action="store", type=str, default="ilp_attack", help="output directory")

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

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=args.jobs)

    alg_dict = {
        "fraudar": do_fraudar,
        "rsd": do_rsd,
        "rev2": do_rev2,
        "fbox": do_fbox,
        "sg": do_sg,
    }

    do_alg = alg_dict[args.alg]

    output_path = Path(f"../res/{args.outdir}/{args.alg}-{args.data}/{args.budget}-{args.frac}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"{output_path} exists! Stop and quit")
        exit(1)

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    created_frauds = [f"usock{a}" for a in range(args.ctotal)]
    created_dummys = [f"udummy{a}" for a in range(2*args.ctotal)]
    existed_frauds = data_gt_df[data_gt_df["label"] == -1]["id"].tolist()

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
    existed_prices = [{u: (1-score[u]) * args.rcost for u in score if u in existed_frauds} for score in scores]

    existed_plans = pool.starmap(
        func=ILPsolve,
        iterable=[(prices, args.rtotal, args.rbudget) for prices in existed_prices],
        chunksize=1,
    )

    G_attacks = pool.starmap(
        func=do_attack,
        iterable=zip(G_list, targets_plans, [created_frauds]*len(G_list), existed_plans, [created_dummys]*len(G_list)),
        chunksize=1,
    )

    print("attack applied")

    scores_final = pool.map(func=do_alg, iterable=G_attacks, chunksize=1)

    # ! only save the users with ground truth, including the socks as well
    output_users = created_frauds + data_gt_df["id"].tolist() + created_dummys
    print(len(data_gt_df["id"].tolist()), len(output_users))
    output_scores = [{u: score[u] for u in score if u in output_users} for score in scores_final]
    with open(output_path, "wb") as fp:
        pickle.dump(output_scores, fp)
