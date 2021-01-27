import itertools
import multiprocessing as mp
from subprocess import Popen

budgets = [100, 200, 300, 400]

algs = ["fraudar", "rev2", "rsd"]
datas = ["alpha", "otc", "amazon", "epinions"][2:3]

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
    pool = mp.Pool(processes=4)
    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
            }
            for a, d, b in itertools.product(algs, datas, budgets)],
        chunksize=1,
    )
