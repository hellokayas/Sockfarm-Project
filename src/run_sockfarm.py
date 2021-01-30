import itertools
import multiprocessing as mp
from subprocess import Popen

budgets = [100, 200, 300, 400]

algs = ["rev2", "rsd", "fraudar"]
datas = ["alpha", "otc", "amazon", "epinions"][3:4]

epochs = {
    "alpha": int(1e2),
    "otc": int(1e2),
    "amazon": int(10),
    "epinions": int(10),
}


def worker(config):
    p = Popen(["python", "sockfarm.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pool = mp.Pool(processes=10)
    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
                "epoch": epochs[d],
            }
            for a, d, b in itertools.product(algs, datas, budgets)],
        chunksize=1,
    )
