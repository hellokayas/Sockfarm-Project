import networkx as nx
import numpy as np
import scipy.sparse.linalg as linalg

from specgreedy.greedy import avgdeg_even, avgdeg_log, avgdeg_sqrt


def sgcompute(G: nx.DiGraph, w_g="sqrt", topk=10, alpha=1.0):
    G_nodes = list(G.nodes)

    sm = nx.to_scipy_sparse_matrix(G, nodelist=G_nodes)
    # es = sm.sum()

    greedy_func = None

    if w_g == 'even':
        greedy_func = avgdeg_even
    elif w_g == 'sqrt':
        greedy_func = avgdeg_sqrt
    else:
        greedy_func = avgdeg_log

    if not w_g:
        # print("max edge weight: {}".format(sm.max()))
        sm = sm > 0
        sm = sm.astype('int')
    # es = sm.sum()
    ms, ns = sm.shape

    opt_k = -1
    opt_density = 0.0
    orgnds = None
    # cans = None
    fin_pms, fin_pns = 0, 0

    k = 0
    decom_n = 0

    start = 3
    step = 3
    isbreak = False
    # t1 = time.time()
    while k < topk:
        print("\ncurrent ks: {}".format(start + decom_n * step))
        U, S, V = linalg.svds(sm.asfptype(), k=start + decom_n * step, which='LM', tol=1e-2)
        U, S, V = U[:, ::-1], S[::-1], V.T[:, ::-1]
        print("lambdas: {}".format(S))
        kth = k
        while kth < start + decom_n * step - 1 and kth < topk:
            if abs(max(U[:, kth])) < abs(min(U[:, kth])):
                U[:, kth] *= -1
            if abs(max(V[:, kth])) < abs(min(V[:, kth])):
                V[:, kth] *= -1
            row_cans = list(np.where(U[:, kth] >= 1.0 / np.sqrt(ms))[0])
            col_cans = list(np.where(V[:, kth] >= 1.0 / np.sqrt(ns))[0])
            if len(row_cans) <= 1 or len(col_cans) <= 1:
                # print("SKIP_ERROR: candidates size: {}".format((len(row_cans), len(col_cans))))
                kth += 1
                k += 1
                continue
            sm_part = sm[row_cans, :][:, col_cans]
            nds_gs, avgsc_gs = greedy_func(sm_part, alpha)
            # print("k_cur: {} size: {}, density: {}".format(kth, (len(nds_gs[0]), len(nds_gs[1])), avgsc_gs))
            kth += 1
            k += 1
            if avgsc_gs > opt_density:
                opt_k, opt_density = kth + 1, avgsc_gs
                (sm_pms, sm_pns) = sm_part.shape
                fin_pms, fin_pns = len(nds_gs[0]), len(nds_gs[1])
                print("+++=== svd tops shape (candidates size): {}".format((sm_pms, sm_pns)))
                print("+++=== size: {}, score: {}\n".format((fin_pms, fin_pns), avgsc_gs))

                row_idx = dict(zip(range(sm_pms), sorted(row_cans)))
                col_idx = dict(zip(range(sm_pns), sorted(col_cans)))
                org_rownds = [row_idx[id] for id in nds_gs[0]]
                org_colnds = [col_idx[id] for id in nds_gs[1]]
                # cans = [row_cans, col_cans]
                orgnds = [org_rownds, org_colnds]

            if 2.0 * opt_density >= S[kth]:  # kth < topk and
                print("k_cur = {},  optimal density: {}, compare: {}".format(kth, opt_density, S[kth]))
                isbreak = True
                break
        if isbreak:
            break
        decom_n += 1

    print("\noptimal k: {}, density: {}".format(opt_k, opt_density))
    # print("total time @ {}s".format(time.time() - t1))
    du = [G_nodes[i] for i in orgnds[0]]
    dp = [G_nodes[i] for i in orgnds[1]]
    return du, dp
