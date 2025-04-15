import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import make_splrep, BSpline, pchip_interpolate
import matplotlib.pyplot as plt
from navtools import RAD2DEG, DEG2RAD
from navtools._navtools_core.frames import lla2ned
from navtools._navtools_core.attitude import quat2euler, euler2quat
from navtools._navtools_core.math import quatdot, quatinv

import sys

sys.path.append("scripts")
from utils.parsers import ParseSturdrLogs, ParseNavSimStates


# dict[str, BSpline]
def ProcessResults(directory: Path, truth: pd.DataFrame, is_array: bool = False):
    res_file = directory / "mc_results.csv"
    if res_file.is_file():
        res = pd.read_csv(res_file)

    else:
        res = pd.DataFrame(
            np.zeros((10, 20)),
            columns=[
                "CNo",
                "J2S",
                "MCn",
                "MCe",
                "MCd",
                "MCvn",
                "MCve",
                "MCvd",
                "MCr",
                "MCp",
                "MCy",
                "KFn",
                "KFe",
                "KFd",
                "KFvn",
                "KFve",
                "KFvd",
                "KFr",
                "KFp",
                "KFy",
            ],
        )
        res["CNo"] = np.array([40, 38, 36, 34, 32, 30, 28, 26, 24, 22])  # , 20])
        res["J2S"] = np.array([22, 24.7, 27, 29.2, 31.3, 33.4, 35.5, 37.5, 39.5, 41.5])  # , 43.5])

        cno_folders = sorted([d for d in directory.iterdir()], reverse=True)
        for kk, cno_fold in enumerate(cno_folders):
            err_list = []
            var_list = []
            for ii in range(30):
                next_folder = cno_fold / f"Run{ii}"
                nav, var, ch = ParseSturdrLogs(next_folder, is_array)
                nav = nav.iloc[:-4]
                var = var.iloc[:-4]
                nav.loc[:, "ToW"] -= nav.loc[0, "ToW"]

                # first, nav results errors need to be time aligned to the truth
                tmp = pd.DataFrame(
                    np.zeros((len(nav), 9)),
                    columns=["lat", "lon", "h", "vn", "ve", "vd", "r", "p", "y"],
                )

                # calculate error
                for jj in range(len(nav)):
                    t = nav.loc[jj, "ToW"]
                    # position error
                    lla_nav = np.array(
                        [nav.loc[jj, "Lat"], nav.loc[jj, "Lon"], nav.loc[jj, "Alt"]],
                        order="F",
                    )
                    lla_true = np.array(
                        [truth["lat"](t), truth["lon"](t), truth["h"](t)], order="F"
                    )
                    ned_err = lla2ned(lla_nav, lla_true)
                    tmp.loc[jj, "lat"] = ned_err[0]
                    tmp.loc[jj, "lon"] = ned_err[1]
                    tmp.loc[jj, "h"] = ned_err[2]

                    # velocity error
                    tmp.loc[jj, "vn"] = nav.loc[jj, "vN"] - truth["vn"](t)
                    tmp.loc[jj, "ve"] = nav.loc[jj, "vE"] - truth["ve"](t)
                    tmp.loc[jj, "vd"] = nav.loc[jj, "vD"] - truth["vd"](t)

                    # attitude error
                    q_nav = np.array(
                        [
                            nav.loc[jj, "qw"],
                            -nav.loc[jj, "qx"],
                            -nav.loc[jj, "qy"],
                            -nav.loc[jj, "qz"],
                        ],
                        order="F",
                    )
                    rpy_true = np.array([truth["r"](t), truth["p"](t), truth["y"](t)], order="F")
                    q_true = euler2quat(rpy_true, True)
                    q_err = quatdot(q_true, q_nav)
                    rpy_err = quat2euler(q_err, True) * RAD2DEG
                    tmp.loc[jj, "r"] = rpy_err[0]
                    tmp.loc[jj, "p"] = rpy_err[1]
                    tmp.loc[jj, "y"] = rpy_err[2]

                # append lists
                err_list.append(tmp)
                var_list.append(var.iloc[-3:])

            # calculate experimental and theoretical variance
            err_df = pd.concat(err_list)
            var_df = pd.concat(var_list)
            res.loc[kk, "MCn"] = np.var(err_df["lat"])
            res.loc[kk, "MCe"] = np.var(err_df["lon"])
            res.loc[kk, "MCd"] = np.var(err_df["h"])
            res.loc[kk, "MCvn"] = np.var(err_df["vn"])
            res.loc[kk, "MCve"] = np.var(err_df["ve"])
            res.loc[kk, "MCvd"] = np.var(err_df["vd"])
            res.loc[kk, "MCr"] = np.var(err_df["r"])
            res.loc[kk, "MCp"] = np.var(err_df["p"])
            res.loc[kk, "MCy"] = np.var(err_df["y"])
            res.loc[kk, "KFn"] = np.mean(var_df["Lat_Var"])
            res.loc[kk, "KFe"] = np.mean(var_df["Lon_Var"])
            res.loc[kk, "KFd"] = np.mean(var_df["Alt_Var"])
            res.loc[kk, "KFvn"] = np.mean(var_df["vN_Var"])
            res.loc[kk, "KFve"] = np.mean(var_df["vE_Var"])
            res.loc[kk, "KFvd"] = np.mean(var_df["vD_Var"])
            res.loc[kk, "KFr"] = np.mean(var_df["Roll_Var"]) * RAD2DEG**2
            res.loc[kk, "KFp"] = np.mean(var_df["Pitch_Var"]) * RAD2DEG**2
            res.loc[kk, "KFy"] = np.mean(var_df["Yaw_Var"]) * RAD2DEG**2

        # save to csv
        res.to_csv(res_file)

    return res


if __name__ == "__main__":
    # for cno in range(22, 42, 2):
    #     nav, var, ch = ParseSturdrLogs(
    #         f"/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim/CNo_{cno}_dB/Run0", True
    #     )
    #     a = 0
    #     for ii in range(10):
    #         a += ch[ii].iloc[-3000:]["CNo"].mean()
    #     a /= 10
    #     print(a)
    # print()

    truth = ParseNavSimStates("data/drone_sim.bin")
    truth = truth.loc[truth["t"] >= 20200]
    truth = truth.reset_index(drop=True)
    truth.loc[:, "t"] -= truth.loc[0, "t"]
    truth.loc[:, "t"] /= 1000.0
    truth_spline = {
        "t": make_splrep(truth["t"].values, truth["t"].values, s=0),
        "lat": make_splrep(truth["t"].values, truth["lat"].values * DEG2RAD, s=0),
        "lon": make_splrep(truth["t"].values, truth["lon"].values * DEG2RAD, s=0),
        "h": make_splrep(truth["t"].values, truth["h"].values, s=0),
        "vn": make_splrep(truth["t"].values, truth["vn"].values, s=0),
        "ve": make_splrep(truth["t"].values, truth["ve"].values, s=0),
        "vd": make_splrep(truth["t"].values, truth["vd"].values, s=0),
        "r": make_splrep(truth["t"].values, truth["r"].values * DEG2RAD, s=0),
        "p": make_splrep(truth["t"].values, truth["p"].values * DEG2RAD, s=0),
        "y": make_splrep(truth["t"].values, truth["y"].values * DEG2RAD, s=0),
    }

    # f, ax = plt.subplots()
    # ax.plot(truth["t"], truth["lat"], linewidth=3)
    # ax.plot(truth_spline["t"](truth["t"]), truth_spline["lat"](truth["t"].values))
    # plt.show()

    directory = Path("/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim-2/")
    res = ProcessResults(directory, truth_spline, True)
    print(res)
