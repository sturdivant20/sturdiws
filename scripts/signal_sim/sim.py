import sys
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
from time import time
from dataclasses import dataclass
from scipy.interpolate import make_splrep, BSpline
from sturdr import SturDR
from navtools import RAD2DEG, DEG2RAD
from navtools._navtools_core.attitude import euler2quat, quat2euler
from navtools._navtools_core.frames import lla2ned
from navtools._navtools_core.math import quatdot, quatinv
from signals import combine_int16_iq_files
from struct import pack

sys.path.append("scripts")
from utils.parsers import ParseNavSimStates, ParseSturdrLogs


@dataclass(slots=True)
class TruthObservables:
    t: BSpline
    lat: BSpline
    lon: BSpline
    h: BSpline
    vn: BSpline
    ve: BSpline
    vd: BSpline
    r: BSpline
    p: BSpline
    y: BSpline
    tR: BSpline


def parse_truth_to_spline(truth_file: Path, init_tR: float = 494998.07) -> TruthObservables:
    # parse truth
    truth = ParseNavSimStates(truth_file)
    tR = np.round(truth["t"].values / 1000.0 + init_tR, 2)

    # convert to splines
    return TruthObservables(
        t=make_splrep(tR, truth["t"].values / 1000.0),
        lat=make_splrep(tR, truth["lat"].values * DEG2RAD),
        lon=make_splrep(tR, truth["lon"].values * DEG2RAD),
        h=make_splrep(tR, truth["h"].values),
        vn=make_splrep(tR, truth["vn"].values, k=5),
        ve=make_splrep(tR, truth["ve"].values, k=5),
        vd=make_splrep(tR, truth["vd"].values, k=5),
        r=make_splrep(tR, truth["r"].values * DEG2RAD, k=5),
        p=make_splrep(tR, truth["p"].values * DEG2RAD, k=5),
        y=make_splrep(tR, np.unwrap(truth["y"].values * DEG2RAD), k=5),
        tR=make_splrep(tR, tR),
    )


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
    t0 = time()
    freeze_support()
    print(f"\u001b[31;1m[sturdiws]\u001b[0m Running Monte Carlo ... ")

    # load config
    yaml_file = Path("config/vt_signal_sim.yaml")
    yaml = YAML()
    with open(yaml_file, "r") as yf:
        conf = yaml.load(yf)

    # grab truth spline
    truth_file = Path("./data/drone_sim.bin")  # "./data/ground_sim.bin"
    truth = parse_truth_to_spline(truth_file, 494998.07)  # 507178.98)

    # loop through each signal power
    indir = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim-downsampled")
    outdir = Path("/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim")
    # indir = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/ground-sim-downsampled")
    # outdir = Path("/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/ground-sim")
    for ii, cno in enumerate(range(20, 42, 2)):
        new_dir = f"CNo_{cno}_dB"

        for jj in tqdm(
            iterable=range(30),
            desc=f"\u001b[31;1m[sturdiws]\u001b[0m C/No = {cno} dB ... ",
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ncols=120,
        ):
            # dump current config
            (outdir / new_dir / f"Run{jj:d}").mkdir(parents=True, exist_ok=True)
            conf["out_folder"] = str(outdir / new_dir)
            conf["scenario"] = f"Run{jj:d}"
            with open(yaml_file, "w") as yf:
                yaml.dump(conf, yf)

            # combine signal and noise files
            pool = Pool(processes=4)
            for kk in range(4):
                sig_file = indir / new_dir / f"Ant-{kk}.bin"
                noise_file = indir / "noise" / f"noise-{jj}-{kk}.bin"
                out_file = Path("./data") / f"SigSim-{kk}.bin"
                pool.apply_async(
                    combine_int16_iq_files, args=(sig_file, noise_file, out_file, 2**28)
                )
            pool.close()
            pool.join()

            # run sturdr
            sdr = SturDR(str(yaml_file))
            sdr.Start()
            del sdr

            # determine errors
            nav, _ = ParseSturdrLogs(outdir / new_dir / f"Run{jj:d}", True)
            with open(outdir / new_dir / f"Run{jj:d}" / "Error_Log.bin", "wb") as fe:
                for kk in range(len(nav)):
                    tR = nav.loc[kk, "tR"]

                    # calculate errors
                    lla_nav = np.array(
                        [nav.loc[kk, "Lat"], nav.loc[kk, "Lon"], nav.loc[kk, "H"]], order="F"
                    )
                    lla_true = np.array(
                        [truth.lat(tR), truth.lon(tR), truth.h(tR)],
                        order="F",
                    )
                    ned_err = lla2ned(lla_nav, lla_true)
                    rpy_true = np.array([truth.r(tR), truth.p(tR), truth.y(tR)], order="F")
                    q_true = euler2quat(rpy_true)
                    q_nav = np.array(
                        [
                            nav.loc[kk, "qw"],
                            nav.loc[kk, "qx"],
                            nav.loc[kk, "qy"],
                            nav.loc[kk, "qz"],
                        ],
                        order="F",
                    )
                    rpy_err = quat2euler(quatdot(q_true, quatinv(q_nav)), True) * RAD2DEG

                    # log errors
                    data = [
                        truth.t(tR),
                        tR,
                        ned_err[0],
                        ned_err[1],
                        ned_err[2],
                        truth.vn(tR) - nav.loc[kk, "vN"],
                        truth.ve(tR) - nav.loc[kk, "vE"],
                        truth.vd(tR) - nav.loc[kk, "vD"],
                        rpy_err[0],
                        rpy_err[1],
                        rpy_err[2],
                    ]
                    fe.write(pack("d" * 11, *data))
                    fe.flush()

    print(f"\u001b[31;1m[sturdiws]\u001b[0m Finished processing in {convert_seconds(time() - t0)}")
