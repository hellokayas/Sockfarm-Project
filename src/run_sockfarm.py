import itertools
import multiprocessing as mp
from subprocess import Popen

budgets = [100]
frac = [0.2]

algs = ["rev2", "rsd", "fraudar"]
datas = ["alpha", "otc", "amazon", "epinions"][:2]


def worker(config):
    p = Popen(["python", "sockfarm.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    pool = mp.Pool(processes=8)
    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
                "frac": f,
            }
            for a, d, b, f in itertools.product(algs, datas, budgets, frac)],
        chunksize=1,
    )
