import itertools
import multiprocessing as mp
from subprocess import Popen

budgets = [100, 200, 300, 400]
frac = [0.7, 0.9]

algs = ["fraudar", "rev2", "rsd", "sg", "fbox"][:4]
datas = ["alpha", "otc", "amazon", "epinions"][:]

outdir = "rtv_attack"


def worker(config):
    p = Popen(["python", "ilp_attack.py", *sum([[f"--{k}", f"{config[k]}"] for k in config], []), ])
    p.wait()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=10)
    pool.map(
        func=worker,
        iterable=[
            {
                "alg": a,
                "data": d,
                "budget": b,
                "frac": f,
                "outdir": outdir,
            }
            for a, d, b, f in itertools.product(algs, datas, budgets, frac)],
        chunksize=1,
    )
