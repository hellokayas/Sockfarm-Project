import numpy as np

import networkx as nx

# algorithm begins here


def rev2compute(G: nx.DiGraph,
                alpha1=1,
                alpha2=1,
                beta1=1,
                beta2=1,
                gamma1=1,
                gamma2=1,
                gamma3=1,
                max_iter=1,
                ):

    for node in G.nodes:
        if "u" in node[0]:
            G.nodes[node]["fairness"] = 1
        else:
            G.nodes[node]["goodness"] = 1

    for edge in G.edges:
        G[edge[0]][edge[1]]["fairness"] = 1

    du = 0
    dp = 0
    dr = 0

    # ! REV2 ITERATIONS START
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
        for node in G.nodes:
            if "p" not in node[0]:
                continue
            currentgvals.append(G.nodes[node]["goodness"])

        # Alternatively, we can use mean here, intead of median
        median_gvals = np.median(currentgvals)

        for node in G.nodes:
            if "p" not in node[0]:
                continue

            inedges = G.in_edges(node,  data=True)
            ftotal = 0.0
            gtotal = 0.0
            for edge in inedges:
                # print(edge)
                gtotal += edge[2]["fairness"]*edge[2]["weight"]
            ftotal += 1.0

            kl_timestamp = 1

            if ftotal > 0.0:
                mean_rating_fairness = (beta1*median_gvals + beta2 * kl_timestamp + gtotal)/(beta1 + beta2 + ftotal)
            else:
                mean_rating_fairness = 0.0

            x = mean_rating_fairness

            if x < -1.0:
                x = -1.0
            if x > 1.0:
                x = 1.0
            dp += abs(G.nodes[node]["goodness"] - x)
            G.nodes[node]["goodness"] = x

        ############################################################

        # ! Update fairness of ratings
        for edge in G.edges:
            rating_distance = 1 - (abs(G.edges[edge]["weight"] - G.nodes[edge[1]]["goodness"])/2.0)

            user_fairness = G.nodes[edge[0]]["fairness"]
            kl_text = 1.0

            x = (gamma2*rating_distance + gamma1*user_fairness + gamma3*kl_text)/(gamma1 + gamma2 + gamma3)

            if x < 0.00:
                x = 0.0
            if x > 1.0:
                x = 1.0

            dr += abs(G.edges[edge]["fairness"] - x)
            # adapt to nx v2
            G.edges[edge]["fairness"] = x

        ############################################################

        currentfvals = []
        for node in G.nodes:
            if "u" not in node[0]:
                continue
            currentfvals.append(G.nodes[node]["fairness"])
            # Alternatively, we can use mean here, intead of median
            median_fvals = np.median(currentfvals)

        # ! update fairness of users
        for node in G.nodes:
            if "u" not in node[0]:
                continue

            outedges = G.out_edges(node, data=True)

            # f = 0
            rating_fairness = []
            for edge in outedges:
                rating_fairness.append(edge[2]["fairness"])

            for x in range(0, alpha1):
                rating_fairness.append(median_fvals)

            kl_timestamp = 1.0

            for x in range(0, alpha2):
                rating_fairness.append(kl_timestamp)

            mean_rating_fairness = np.mean(rating_fairness)

            x = mean_rating_fairness  # *(kl_timestamp)
            if x < 0.00:
                x = 0.0
            if x > 1.0:
                x = 1.0

            du += abs(G.nodes[node]["fairness"] - x)
            G.nodes[node]["fairness"] = x

        if du < 0.1 and dp < 0.1 and dr < 0.1:
            break

    scores = {n: G.nodes[n]["fairness"] for n in G.nodes if n[0] == "u"}
    return scores
