import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("scripts")
from utils.parsers import ParseSturdrLogs, ParseCorrelatorSimLogs
from utils.plotters import MyWindow, MatplotlibWidget
from navtools import RAD2DEG, LIGHT_SPEED

SIM = "drone"  # "ground"
MODEL = "Signal"  # "Correlator"


def CalcMCVar(directory: Path | str, is_array: bool = False):
    res_file = directory / "mc_results.csv"
    # if res_file.is_file():
    #     res = pd.read_csv(res_file)

    # else:
    res = pd.DataFrame(
        np.zeros((11, 24)),
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
            "MCcb",
            "MCcd",
            "KFn",
            "KFe",
            "KFd",
            "KFvn",
            "KFve",
            "KFvd",
            "KFr",
            "KFp",
            "KFy",
            "KFcb",
            "KFcd",
        ],
    )
    res["CNo"] = np.arange(20, 42, 2)
    res["J2S"] = np.arange(43, 21, -2.0)

    cno_folders = sorted([d for d in directory.iterdir()])
    for kk, cno_fold in enumerate(cno_folders):
        if cno_fold.is_file():
            continue
        err_list = []
        var_list = []
        for ii in range(30):
            next_folder = cno_fold / f"Run{ii}"
            if MODEL == "Signal":
                nav, err, ch = ParseSturdrLogs(next_folder, is_array, True)
            else:
                nav, err, ch = ParseCorrelatorSimLogs(next_folder, is_array)
            err["Bias"] = nav["Bias"]
            err["Drift"] = nav["Drift"]
            if SIM == "drone":
                err_list.append(err.loc[(err["t"] > 30.0)])
                var_list.append(nav.loc[(err["t"] > 105.0)])
            else:
                err_list.append(err.loc[(err["t"] > 30.0) & (err["t"] < 450.0)])
                var_list.append(nav.loc[(err["t"] > 440.0) & (err["t"] < 450.0)])
        if MODEL == "Signal":
            err_mean = sum(err_list) / 30
        else:
            err_mean = sum(err_list) / 100
        err_df = pd.concat(err_list) - err_mean
        var_df = pd.concat(var_list)

        res.loc[kk, "MCn"] = np.var(err_df["N"])
        res.loc[kk, "MCe"] = np.var(err_df["E"])
        res.loc[kk, "MCd"] = np.var(err_df["D"])
        res.loc[kk, "MCvn"] = np.var(err_df["vN"])
        res.loc[kk, "MCve"] = np.var(err_df["vE"])
        res.loc[kk, "MCvd"] = np.var(err_df["vD"])
        res.loc[kk, "MCr"] = np.var(err_df["Roll"])
        res.loc[kk, "MCp"] = np.var(err_df["Pitch"])
        res.loc[kk, "MCy"] = np.var(err_df["Yaw"])
        res.loc[kk, "MCcb"] = np.var(err_df["Bias"])
        res.loc[kk, "MCcd"] = np.var(err_df["Drift"])

        res.loc[kk, "KFn"] = np.mean(var_df["P0"])
        res.loc[kk, "KFe"] = np.mean(var_df["P1"])
        res.loc[kk, "KFd"] = np.mean(var_df["P2"])
        res.loc[kk, "KFvn"] = np.mean(var_df["P3"])
        res.loc[kk, "KFve"] = np.mean(var_df["P4"])
        res.loc[kk, "KFvd"] = np.mean(var_df["P5"])
        res.loc[kk, "KFr"] = np.mean(var_df["P6"])
        res.loc[kk, "KFp"] = np.mean(var_df["P7"])
        res.loc[kk, "KFy"] = np.mean(var_df["P8"])
        res.loc[kk, "KFcb"] = np.mean(var_df["P9"])
        res.loc[kk, "KFcd"] = np.mean(var_df["P10"])
        if MODEL == "Signal":
            res.loc[kk, "KFr"] *= RAD2DEG**2
            res.loc[kk, "KFp"] *= RAD2DEG**2
            res.loc[kk, "KFy"] *= RAD2DEG**2
            res.loc[kk, "KFcb"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk, "KFcd"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk, "MCcb"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk, "MCcd"] *= (1e9 / LIGHT_SPEED) ** 2

    # save to csv
    res.to_csv(res_file)
    return res


if __name__ == "__main__":
    dir = Path(f"/mnt/f/Thesis-Data/{MODEL}-Sim/{SIM}-sim")
    res = CalcMCVar(dir, True)
