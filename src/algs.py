import fraudar
import networkx as nx
import numpy as np
import rsd


class fraudar_alg():
    def __init__(
            self,
            G: nx.DiGraph,
            blocks=8
    ):
        self.graph = fraudar.ReviewGraph(blocks=8, algo=fraudar.aveDegree)
        self.reviewers_dict = {n: self.graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
        self.products_dict = {n: self.graph.new_product(n) for n in G.nodes if n.startswith("p")}

        # ! the rating must in range [0, 1]
        for e in G.edges:
            self.graph.add_review(self.reviewers_dict[e[0]], self.products_dict[e[1]], (G.edges[e]["rating"] + 1)/2)

    def update(self, max_iter=5):
        # * max_iter is a placeholder for fraudar
        self.graph.update()

    def add_review(self, user=None, prod=None, rating=1):
        self.graph.add_review(self.reviewers_dict[user], self.products_dict[prod], rating)

    def get_score(self):
        # ! higher means anormalous
        scores = {r.name: r.anomalous_score for r in self.graph.reviewers}
        return scores


class rsd_alg():
    def __init__(
            self,
            G: nx.DiGraph,
            theta=0.25,
    ):

        self.graph = rsd.ReviewGraph(theta)
        self.reviewers_dict = {n: self.graph.new_reviewer(n) for n in G.nodes if n.startswith("u")}
        self.products_dict = {n: self.graph.new_product(n) for n in G.nodes if n.startswith("p")}

        # ! the rating must in range [0, 1]
        for e in G.edges:
            self.graph.add_review(self.reviewers_dict[e[0]], self.products_dict[e[1]], (G.edges[e]["rating"] + 1)/2)

    def update(self, max_iter=10, threshold=1e-3):
        for it in range(max_iter):
            diff = self.graph.update()
            if diff < threshold:
                break

    def add_review(self, user=None, prod=None, rating=1):
        self.graph.add_review(self.reviewers_dict[user], self.products_dict[prod], rating)

    def get_score(self):
        # ! higher means anormalous
        scores = {r.name: r.anomalous_score for r in self.graph.reviewers}
        return scores


class rev2_alg():
    def __init__(
        self,
        G: nx.DiGraph,
        alpha1=1,
        alpha2=1,
        beta1=1,
        beta2=1,
        gamma1=1,
        gamma2=1,
        gamma3=1,
    ):

        self.G = G.copy()
        for e in self.G.edges:
            self.G.edges[e]["weight"] = (self.G.edges[e]["rating"] + 1)/2

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        for node in self.G.nodes:
            if "u" in node[0]:
                self.G.nodes[node]["fairness"] = 1
            else:
                self.G.nodes[node]["goodness"] = 1

        for edge in self.G.edges:
            self.G[edge[0]][edge[1]]["fairness"] = 1

    def get_score(self):
        # ! higher means anormalous
        scores = {n: 1-self.G.nodes[n]["fairness"] for n in self.G.nodes if n[0] == "u"}
        return scores

    def add_review(self, user=None, prod=None, rating=1):
        assert user in self.G.nodes
        assert prod in self.G.nodes
        self.G.add_edge(user, prod, weight=rating, fairness=1)

    def update(self, max_iter=5):
        du = 0
        dp = 0
        dr = 0

        for i in range(max_iter):
            # print('-----------------')
            # print("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)" %
            #       (iter, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
            # print(f"{i}> du: {du}, dp: {dp}, dr: {dr}")
            if np.isnan(du) or np.isnan(dp) or np.isnan(dr):
                break

            du = 0
            dp = 0
            dr = 0

            ############################################################

            # ! Update goodness of product
            currentgvals = []
            for node in self.G.nodes:
                if "p" not in node[0]:
                    continue
                currentgvals.append(self.G.nodes[node]["goodness"])

            # Alternatively, we can use mean here, intead of median
            median_gvals = np.median(currentgvals)

            for node in self.G.nodes:
                if "p" not in node[0]:
                    continue

                inedges = self.G.in_edges(node,  data=True)
                ftotal = 0.0
                gtotal = 0.0
                for edge in inedges:
                    # print(edge)
                    gtotal += edge[2]["fairness"]*edge[2]["weight"]
                ftotal += 1.0

                kl_timestamp = 1

                if ftotal > 0.0:
                    mean_rating_fairness = (self.beta1*median_gvals + self.beta2 *
                                            kl_timestamp + gtotal)/(self.beta1 + self.beta2 + ftotal)
                else:
                    mean_rating_fairness = 0.0

                x = mean_rating_fairness

                if x < -1.0:
                    x = -1.0
                if x > 1.0:
                    x = 1.0
                dp += abs(self.G.nodes[node]["goodness"] - x)
                self.G.nodes[node]["goodness"] = x

            ############################################################

            # ! Update fairness of ratings
            for edge in self.G.edges:
                rating_distance = 1 - (abs(self.G.edges[edge]["weight"] - self.G.nodes[edge[1]]["goodness"])/2.0)

                # print(f"{edge[0]} {self.G.nodes[edge[0]]}")
                user_fairness = self.G.nodes[edge[0]]["fairness"]
                kl_text = 1.0

                x = (self.gamma2*rating_distance + self.gamma1*user_fairness +
                     self.gamma3*kl_text)/(self.gamma1+self.gamma2 + self.gamma3)

                if x < 0.00:
                    x = 0.0
                if x > 1.0:
                    x = 1.0

                dr += abs(self.G.edges[edge]["fairness"] - x)
                # adapt to nx v2
                self.G.edges[edge]["fairness"] = x

            ############################################################

            currentfvals = []
            for node in self.G.nodes:
                if "u" not in node[0]:
                    continue
                currentfvals.append(self.G.nodes[node]["fairness"])
                # Alternatively, we can use mean here, intead of median
                median_fvals = np.median(currentfvals)

            # ! update fairness of users
            for node in self.G.nodes:
                if "u" not in node[0]:
                    continue

                outedges = self.G.out_edges(node, data=True)

                # f = 0
                rating_fairness = []
                for edge in outedges:
                    rating_fairness.append(edge[2]["fairness"])

                for x in range(0, self.alpha1):
                    rating_fairness.append(median_fvals)

                kl_timestamp = 1.0

                for x in range(0, self.alpha2):
                    rating_fairness.append(kl_timestamp)

                mean_rating_fairness = np.mean(rating_fairness)

                x = mean_rating_fairness  # *(kl_timestamp)
                if x < 0.00:
                    x = 0.0
                if x > 1.0:
                    x = 1.0

                du += abs(self.G.nodes[node]["fairness"] - x)
                self.G.nodes[node]["fairness"] = x

            if du < 0.1 and dp < 0.1 and dr < 0.1:
                break
