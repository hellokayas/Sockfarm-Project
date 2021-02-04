import itertools
import multiprocessing as mp
from subprocess import Popen
import argparse

budgets = [100, 200, 300, 400]

algs = ["rev2", "rsd", "fraudar"]
datas = ["alpha", "otc", "amazon", "epinions"][2:3]

epochs = {
    "alpha": int(1e2),
    "otc": int(1e2),
    "amazon": int(50),
    "epinions": int(10),
}

req = int(1e4)
budgets = [b*100 for b in budgets]


def worker(config):
    p = Popen(["python", "sockfarm.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run sockfarm large")
    parser.add_argument("--mode", action="store", type=str, choices=["single", "double"], default="single")
    args = parser.parse_args()
    print(args)

    outdir = f"sockfarm_large_{args.mode}"
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
                "epoch": epochs[d],
                "req": req,
                "outdir": outdir,
                "layers": layers,
                "policy": "sockfarm",
            }
            for a, d, b in itertools.product(algs, datas, budgets)],
        chunksize=1,
    )
