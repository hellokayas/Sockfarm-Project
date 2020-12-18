import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import fraud_eagle as feagle
import fraudar
import networkx as nx
import numpy as np
import pandas as pd
import rsd

from rev2 import rev2compute
from utils import load_data, split_data_by_time


def build_nx(data_nw_df):
    G = nx.from_pandas_edgelist(
        data_nw_df,
        source="src",
        target="dest",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    return G


def do_fraudar(G):
    # blocks means patterns, a hyper-param for fraudar
    graph = fraudar.ReviewGraph(blocks=8, algo=fraudar.aveDegree)
    reviewers = {n: graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
    products = {n: graph.new_product(n) for n in G.nodes if n.startswith("p")}

    # ! the rating must in range [0, 1]
    for e in G.edges:
        graph.add_review(reviewers[e[0]], products[e[1]], (G.edges[e]["rating"] + 1)/2)

    graph.update()

    # ! higher means anormalous
    scores = {r.name: r.anomalous_score for r in graph.reviewers}
    return scores


def do_rsd(G):
    theta = 0.25
    graph = rsd.ReviewGraph(theta)
    # blocks means patterns, a hyper-param for fraudar
    reviewers = {n: graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
    products = {n: graph.new_product(n) for n in G.nodes if n.startswith("p")}

    # ! the rating must in range [0, 1]
    for e in G.edges:
        graph.add_review(reviewers[e[0]], products[e[1]], (G.edges[e]["rating"] + 1)/2)

    for it in range(10):
        diff = graph.update()
        if diff < 1e-3:
            break

    # ! higher means anormalous
    scores = {r.name: r.anomalous_score for r in graph.reviewers}
    return scores


def do_feagle(G):
    epsilon = 0.25
    graph = feagle.ReviewGraph(epsilon)
    # blocks means patterns, a hyper-param for fraudar
    reviewers = {n: graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
    products = {n: graph.new_product(n) for n in G.nodes if n.startswith("p")}

    # ! the rating must in range [0, 1]
    for e in G.edges:
        graph.add_review(reviewers[e[0]], products[e[1]], (G.edges[e]["rating"] + 1)/2)

    for it in range(10):
        diff = graph.update()
        if diff < 1e-3:
            break

    # ! higher means anormalous
    scores = {r.name: r.anomalous_score for r in graph.reviewers}
    return scores


def do_rev2(G):
    G = G.copy()
    for e in G.edges:
        G.edges[e]["weight"] = G.edges[e]["rating"]
    # ! lower means anormalous
    scores = rev2compute(G, max_iter=15)
    return scores


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

    data_nw_df, data_gt_df = load_data(data_name=args.data)

    longest = data_nw_df["src"].map(lambda x: len(x)).max()
    socks = [f"u{a}" for a in np.random.randint(low=10**longest, high=10**(longest+1), size=args.acc)]

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
    output_path = Path(f"../res/naive_attack/{args.alg}-{args.data}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fp:
        pickle.dump(output_scores, fp)
