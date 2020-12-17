import fraudar
import fraud_eagle as feagle
import rsd
import argparse
import pandas as pd
import networkx as nx
from datetime import datetime
from rev2 import rev2compute


def load_data(data_name):
    data_nw_df = pd.read_csv(f"../rev2data/{data_name}/{data_name}_network.csv", header=None,
                             names=["src", "dest", "rating", "timestamp"], parse_dates=[3], infer_datetime_format=True)
    data_gt_df = pd.read_csv(f"../rev2data/{data_name}/{data_name}_gt.csv", header=None, names=["id", "label"])

    if data_name != "epinions":
        data_nw_df["timestamp"] = data_nw_df["timestamp"].astype(int).apply(datetime.fromtimestamp)

    data_nw_df["src"] = "u" + data_nw_df["src"].astype(str)
    data_nw_df["dest"] = "p" + data_nw_df["dest"].astype(str)
    data_nw_df["rating"] = (data_nw_df["rating"] - data_nw_df["rating"].min()) / \
        (data_nw_df["rating"].max() - data_nw_df["rating"].min()) * 2 - 1

    return data_nw_df, data_gt_df


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
    scores = rev2compute(G, max_iter=15)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random attacks on data")
    parser.add_argument("--alg", action="store", type=str, choices=["fraudar", "feagle", "rsd", "rev2"])
    parser.add_argument("--data", action="store", type=str, choices=["alpha", "otc", "amazon", "epinions"])

    args = parser.parse_args()
    print(args)

    data_nw_df, data_gt_df = load_data(data_name=args.data)
    G = build_nx(data_nw_df)

    if args.alg == "fraudar":
        scores = do_fraudar(G)
    elif args.alg == "feagle":
        socres = do_feagle(G)
    elif args.alg == "rsd":
        scores = do_rsd(G)
    elif args.alg == "rev2":
        scores = do_rev2(G)
