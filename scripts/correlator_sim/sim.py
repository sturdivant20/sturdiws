# import numpy as np
import time
from numpy.random import randint
from multiprocessing import Pool, freeze_support, cpu_count
from tqdm import tqdm
from vt import vt, vt_array


def run(args):
    vt_array(*args)


if __name__ == "__main__":
    # np.set_printoptions(precision=4, linewidth=100)
    t0 = time.time()
    freeze_support()
    print(f"\u001b[31;1m[sturdiws]\u001b[0m Running Monte Carlo ... ")

    vt_array("config/vt_correlator_sim.yaml", 33.4, 0)

    # with Pool(processes=cpu_count()) as p:

    #     # loop through cno from 40-20 dB-Hz
    #     for j2s in [22, 24.7, 27, 29.2, 31.3, 33.4, 35.5, 37.5, 39.5, 41.5, 43.5]:
    #         desc = f"\u001b[31;1m[sturdiws]\u001b[0m J2S = {j2s} dB ... "
    #         args = ["config/vt_correlator_sim.yaml", j2s]

    #         # 100 monte carlo runs
    #         for _ in tqdm(
    #             p.imap(run, [args + [i, randint(0, 2**63 - 1)] for i in range(100)]),
    #             total=100,
    #             desc=desc,
    #             ascii=".>#",
    #             bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    #             ncols=120,
    #         ):
    #             pass

    print(f"\u001b[31;1m[sturdiws]\u001b[0m Finised processing in {(time.time() - t0):.3f} seconds")
