from numpy import arange
from time import time
from multiprocessing import Pool, freeze_support, cpu_count
from tqdm import tqdm
from pathlib import Path
from ruamel.yaml import YAML
from secrets import randbits
from vt import VectorTrackingSim


def run(file: str | Path, run_index: int, seed: int):
    # print(f"{file} -- {run_index} -- {seed}")
    vt = VectorTrackingSim(file, run_index, seed)
    vt.Run()
    return


def update_progress(_):
    pbar.update(1)
    return


def convert_seconds(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int(seconds % 1 * 1000)
    seconds = int(seconds)
    milliseconds -= seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


if __name__ == "__main__":
    # np.set_printoptions(precision=6, linewidth=100)
    t0 = time()
    freeze_support()
    print(f"\u001b[31;1m[sturdiws]\u001b[0m Running Monte Carlo ... ")

    # load config
    yaml = YAML()
    yaml_file = Path("config/vt_correlator_sim.yaml")
    with open(yaml_file, "r") as yf:
        conf = yaml.load(yf)

    # edit global settings
    conf["data_file"] = "data/drone_sim.bin"
    conf["ephem_file"] = "data/gps_skydel_2024_08_23.bin"
    conf["scenario"] = "drone-sim"
    conf["init_tow"] = 494998.07
    conf["week"] = 2328
    # conf["data_file"] = "data/ground_sim.bin"
    # conf["ephem_file"] = "data/gps_skydel_2025_02_07.bin"
    # conf["scenario"] = "ground-sim"
    # conf["init_tow"] = 507260.0
    # conf["week"] = 2352
    # print(f"\u001b[31;1m[sturdiws]\u001b[0m Scenario: {conf["scenario"]} ... ")

    # ensure 100 different seeds
    unique_seeds = set()
    while len(unique_seeds) < 100:
        unique_seeds.add(randbits(128))
    unique_seeds = list(unique_seeds)

    # loop through cno from 40-20 dB-Hz
    for j2s, cno in zip(arange(43, 21, -2), arange(20, 42, 2)):
        desc = f"\u001b[31;1m[sturdiws]\u001b[0m C/No = {cno} | J/S = {j2s:.1f} dB ... "
        args = [str(yaml_file)]

        # update config file
        conf["cno"] = int(cno)
        conf["j2s"] = float(j2s)
        with open(yaml_file, "w") as yf:
            yaml.dump(conf, yf)

        with tqdm(
            total=100,
            desc=desc,
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ncols=120,
        ) as pbar:
            pool = Pool(processes=cpu_count())
            for kk in range(100):
                pool.apply_async(
                    run, args=(yaml_file, kk, unique_seeds[kk]), callback=update_progress
                )
            pool.close()
            pool.join()

    print(f"\u001b[31;1m[sturdiws]\u001b[0m Finished processing in {convert_seconds(time() - t0)}")
