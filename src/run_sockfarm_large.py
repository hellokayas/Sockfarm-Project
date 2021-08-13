import itertools
import multiprocessing as mp
from subprocess import Popen
import argparse

budgets = [100, 200, 300, 400]

algs = ["rev2", "rsd", "fraudar", "sg"]
datas = ["alpha", "otc", "amazon", "epinions"][:2]

epochs = {
    "alpha": int(1e2),
    "otc": int(1e2),
    "amazon": int(10),
    "epinions": int(20),
}

req = int(1e3)
budgets = [b*10 for b in budgets]
ccost = 25


def worker(config):
    p = Popen(["python", "sockfarm.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run sockfarm large")
    parser.add_argument("--mode", action="store", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--sub", action="store", type=float, default=1.)
    args = parser.parse_args()
    print(args)

    outdir = f"sockfarm_large_{args.mode}_{args.sub:.2}"
    layers = 1 if args.mode == "single" else 2

    mp.set_start_method("spawn")
    pool = mp.Pool(processes=8)
    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
                "epoch": int(epochs[d] * args.sub),
                "req": req,
                "outdir": outdir,
                "layers": layers,
                "policy": "sockfarm",
                "ccost": ccost,
            }
            for a, d, b in itertools.product(algs, datas, budgets)],
        chunksize=1,
    )
