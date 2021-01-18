import fraud_eagle as feagle
import fraudar
import networkx as nx
import rsd

from rev2 import rev2compute


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


def do_rsd(G: nx.DiGraph):
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
    rev2res = rev2compute(G, max_iter=8)
    # ! higher means anormalous
    scores = {u: 1-rev2res[u] for u in rev2res}
    return scores
