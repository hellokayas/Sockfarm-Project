import fraud_eagle as feagle
import fraudar
import networkx as nx
import rsd

from rev2 import rev2compute
from UGFraud.Detector.fBox import fBox
from sgcompute import sgcompute


def do_fraudar(G: nx.DiGraph):
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


def do_rsd(G: nx.DiGraph, max_iter: int = 3):
    theta = 0.25
    graph = rsd.ReviewGraph(theta)
    # blocks means patterns, a hyper-param for fraudar
    reviewers = {n: graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
    products = {n: graph.new_product(n) for n in G.nodes if n.startswith("p")}

    # ! the rating must in range [0, 1]
    for e in G.edges:
        graph.add_review(reviewers[e[0]], products[e[1]], (G.edges[e]["rating"] + 1)/2)

    for it in range(3):
        diff = graph.update()
        if diff < 1e-3:
            break

    # ! higher means anormalous
    scores = {r.name: r.anomalous_score for r in graph.reviewers}
    return scores


def do_feagle(G: nx.DiGraph):
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


def do_rev2(G: nx.DiGraph):
    G = G.copy()
    for e in G.edges:
        G.edges[e]["weight"] = (G.edges[e]["rating"] + 1)/2
    # ! lower means anormalous for rev2
    rev2res = rev2compute(G, max_iter=3)
    # ! higher means anormalous
    scores = {u: 1-rev2res[u] for u in rev2res}
    return scores


def do_fbox(G: nx.DiGraph):
    G = G.copy()
    model = fBox(G)
    detected_user, detected_prod = model.run(tau=20, k=50)
    # * summarize the detected users
    du = set(sum([detected_user[d] for d in detected_user], []))
    # ! 1 means anormalous 0 means begnin
    scores = {u: 1 if u in du else 0 for u in G}
    return scores


def do_sg(G: nx.DiGraph):
    G = G.copy()
    du, _ = sgcompute(G)
    # * summarize the detected users
    # ! 1 means anormalous 0 means begnin
    scores = {u: 1 if u in du else 0 for u in G}
    return scores
