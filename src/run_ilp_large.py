import itertools
import multiprocessing as mp
from subprocess import Popen

budgets = [100, 200, 300, 400]
frac = [0.0, 0.2, 0.4, 0.6, 0.8]

algs = ["fraudar", "rev2", "rsd", "sg", "fbox"][:4]
datas = ["alpha", "otc", "amazon", "epinions"][:4]

req = int(1e3)
budgets = [b*10 for b in budgets]
ccost = 40
outdir = "ilp_large"


def worker(config):
    p = Popen(["python", "ilp_attack.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=4)

    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
                "frac": f,
                "req": req,
                "outdir": outdir,
                "ccost": ccost,
            }
            for a, d, b, f in itertools.product(algs, datas, budgets, frac)],
        chunksize=1,
    )
