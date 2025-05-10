import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pickle import load
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append("scripts/utils")
sys.path.append("scripts/correlator_sim")
from parsers import ParseSturdrLogs, ParseCorrelatorSimLogs
from vt import TruthObservables
from navtools import RAD2DEG, DEG2RAD, LIGHT_SPEED, TWO_PI, PI
from navtools._navtools_core.attitude import euler2dcm
from satutils import GPS_L1_FREQUENCY, GPS_CA_CODE_RATE
from sturdr._sturdr_core.discriminator import *

SIM = "drone"  # "ground"
MODEL = "Correlator"  # "Signal"
LAMBDA = LIGHT_SPEED / (TWO_PI * GPS_L1_FREQUENCY)
BETA = LIGHT_SPEED / GPS_CA_CODE_RATE


def CalcTrueSpatialPhase(truth_file: str | Path, tR: np.ndarray):
    with open(truth_file, "rb") as file:
        truth = load(file)
    L1 = len(tR)
    L2 = len(truth.az)
    spatial_phase = np.zeros((4, L1, L2), order="F")
    ant_xyz = np.array(
        [[0.0, 0.09514, 0.0, 0.09514], [0.0, 0.0, -0.09514, -0.09514], [0.0, 0.0, 0.0, 0.0]],
        order="F",
    )
    u = np.zeros(3, order="F")
    rpy = np.zeros(3, order="F")
    for kk in range(L1):
        for ii in range(L2):
            az = truth.az[ii][0](tR[kk])
            el = truth.el[ii][0](tR[kk])
            # if kk == 0:
            #     print(f"{ii}: az={RAD2DEG*az}, el={RAD2DEG*el}")
            u[0] = np.cos(az) * np.cos(el)
            u[1] = np.sin(az) * np.cos(el)
            u[2] = -np.sin(el)
            rpy[0] = truth.r(tR[kk])
            rpy[1] = truth.p(tR[kk])
            rpy[2] = truth.y(tR[kk])
            C_b_n = euler2dcm(rpy)
            spatial_phase[:, kk, ii] = ((C_b_n @ ant_xyz).T @ u) / LAMBDA

    spatial_phase = np.fmod(spatial_phase + PI, TWO_PI) - PI
    spatial_phase[spatial_phase > PI] -= TWO_PI
    spatial_phase[spatial_phase < -PI] += TWO_PI
    return spatial_phase


def CalcEstDisc(channels: list[pd.DataFrame], tR: np.ndarray, is_array: bool = True):
    # limit channels to valid range
    for ii, ch in enumerate(channels):
        if MODEL == "Correlator":
            if SIM == "drone":
                channels[ii] = ch.loc[(ch["t"] > 30)]
            else:
                channels[ii] = ch.loc[(ch["t"] > 30) & (ch["t"] < 445)]
        else:
            if SIM == "drone":
                channels[ii] = ch.loc[(ch["t"] > 30000)]
            else:
                channels[ii] = ch.loc[(ch["t"] > 30000) & (ch["t"] < 445000)]
    # for ii, ch in enumerate(channels):
    #     if SIM == "drone":
    #         channels[ii] = ch.loc[(ch["ToW"] > tR[0] - 0.069)]
    #     else:
    #         channels[ii] = ch.loc[(ch["ToW"] > tR[0] - 0.069) & (ch["ToW"] < tR[-1] - 0.069)]

    L1 = min([len(channels[ii]) for ii in range(len(channels))])
    channels = [channels[ii][:L1].reset_index(drop=True) for ii in range(len(channels))]
    L2 = len(channels)

    psr = np.zeros((L1, L2), order="F")
    psrdot = np.zeros((L1, L2), order="F")
    dP = np.zeros(4, order="F")
    spatial_phase = np.zeros((4, L1, L2), order="F")
    for kk in range(L1):
        for ii in range(L2):
            psr[kk, ii] = BETA * DllNneml2(
                channels[ii].loc[kk, "IE"],
                channels[ii].loc[kk, "QE"],
                channels[ii].loc[kk, "IL"],
                channels[ii].loc[kk, "QL"],
            )
            psrdot[kk, ii] = LAMBDA * FllAtan2(
                channels[ii].loc[kk, "IP1"],
                channels[ii].loc[kk, "QP1"],
                channels[ii].loc[kk, "IP2"],
                channels[ii].loc[kk, "QP2"],
                0.02,
            )
            if is_array:
                for jj in range(4):
                    dP[jj] = PllAtan2(
                        channels[ii].loc[kk, f"IP_reg_{jj}"], channels[ii].loc[kk, f"QP_reg_{jj}"]
                    )
                spatial_phase[:, kk, ii] = np.fmod(dP - dP[0] + PI, TWO_PI) - PI

    spatial_phase[spatial_phase > PI] -= TWO_PI
    spatial_phase[spatial_phase < -PI] += TWO_PI
    return psr, psrdot, spatial_phase


def func(ii, cno_fold, is_array, truth_file, n_ch):
    next_folder = cno_fold / f"Run{ii}"

    # use correct parser
    if MODEL == "Signal":
        nav, err, ch = ParseSturdrLogs(next_folder, is_array, True)
        err["Bias"] = nav["Bias"]
        err["Drift"] = nav["Drift"]
        if SIM == "drone":
            svid = [5, 10, 13, 15, 18, 23, 24, 27, 29, 32]
        else:
            svid = [1, 2, 3, 6, 11, 14, 17, 19, 22, 24, 30]
        idx = []
        for ii in range(len(ch)):
            idx.append(svid.index(ch[ii].loc[0, "SVID"]))
        ch = [ch[ii] for ii in np.argsort(idx)]
    else:
        nav, err, ch = ParseCorrelatorSimLogs(next_folder, is_array)
        if SIM == "ground":
            ch = [ch[ii] for ii in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]]
        # for ii in range(len(ch)):
        #     print(
        #         f"{ii}: az={(ch[ii].loc[1500,"az"]+180)%360-180}, el={(ch[ii].loc[1500,"el"]+180)%360-180}"
        #     )
        # print("-----")

    # use only valid data
    if SIM == "drone":
        idx = (err["t"] > 30.0).values
        err_out = err.loc[(err["t"] > 30.0)]
        var_out = nav.loc[(err["t"] > 105.0)]
    else:
        idx = ((err["t"] > 30.0) & (err["t"] < 445.0)).values
        err_out = err.loc[(err["t"] > 30.0) & (err["t"] < 445.0)]
        var_out = nav.loc[(err["t"] > 435.0) & (err["t"] < 445.0)]

    # --- calculate channel errors ---
    if is_array:
        est_psr, est_psrdot, est_sp = CalcEstDisc(ch, nav["tR"].loc[idx].values, is_array)
        true_sp = CalcTrueSpatialPhase(truth_file, nav["tR"].loc[idx].values)
        if est_sp.shape[1] != true_sp.shape[1]:
            if est_sp.shape[1] > true_sp.shape[1]:
                est_sp = est_sp[:, : true_sp.shape[1], :]
            else:
                true_sp = true_sp[:, : est_sp.shape[1], :]
        err_sp = true_sp - est_sp
        err_sp = np.fmod(err_sp + PI, TWO_PI) - PI
        err_sp[err_sp > PI] -= TWO_PI
        err_sp[err_sp < -PI] += TWO_PI
    else:
        est_psr, est_psrdot, err_sp = CalcEstDisc(ch, nav["tR"].loc[idx].values, is_array)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # from cycler import cycler

    # color_cycle = list(sns.color_palette().as_hex())
    # color_cycle.append("#100c08")  # "#a2e3b8"
    # color_cycle = cycler("color", color_cycle)
    # f, ax = plt.subplots()
    # ax.set_prop_cycle(color_cycle)
    # [ax.plot(err_sp[1, :, ii]) for ii in range(n_ch)]
    # plt.show()

    ch_out = []
    for i in range(n_ch):
        ch_out.append(
            pd.DataFrame(
                np.asarray(
                    [
                        est_psr[:, i],
                        est_psrdot[:, i],
                        err_sp[1, :, i],
                        err_sp[2, :, i],
                        err_sp[3, :, i],
                    ]
                ).T,
                columns=["dR", "dRR", "dP1", "dP2", "dP3"],
            )
        )
    return err_out, var_out, ch_out


def CalcMCStats(directory: Path | str, is_array: bool = False, n_ch: int = 10):
    # truth_file = dir / "splines/Run1.bin"
    # truth_file = f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{SIM}-sim/splines/Run99.bin"
    truth_file = f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{SIM}-sim/splines/Run1.bin"
    res = pd.DataFrame(
        np.zeros((33, 14)),
        columns=[
            "Model",
            "Type",
            "CNo",
            "n",
            "e",
            "d",
            "vn",
            "ve",
            "vd",
            "r",
            "p",
            "y",
            "cb",
            "cd",
        ],
    )
    res["Model"] = res["Model"].astype(str)
    res["Type"] = res["Type"].astype(str)
    ch_res = [
        pd.DataFrame(
            np.zeros((33, 8)),
            columns=[
                "Model",
                "Type",
                "CNo",
                "dR",
                "dRR",
                "dP1",
                "dP2",
                "dP3",
            ],
        )
        for _ in range(n_ch)
    ]

    res.loc[0:10, "CNo"] = np.arange(20, 42, 2)
    res.loc[11:21, "CNo"] = res.loc[:10, "CNo"].values
    res.loc[22:33, "CNo"] = res.loc[:10, "CNo"].values
    res.loc[:, "Model"] = MODEL
    res.loc[0:10, "Type"] = "Analytical"
    res.loc[11:21, "Type"] = "Empirical"
    res.loc[22:33, "Type"] = "RMSE"
    for c in ch_res:
        c["CNo"] = res["CNo"].values
        c["Model"] = res["Model"].values
        c["Type"] = res["Type"].values

    cno_folders = sorted([d for d in directory.iterdir() if not d.is_file()])
    # if MODEL == "Correlator":
    #     cno_folders = cno_folders[:-1]
    # cno_folders = sorted([d for d in directory.iterdir() if not d.is_file()], reverse=True)
    # cno_folders = cno_folders[1:]
    n_runs = len(list(cno_folders[0].glob("*")))
    for kk, cno_fold in enumerate(cno_folders):
        with Pool(processes=20) as pool:
            args = [[ii, cno_fold, is_array, truth_file, n_ch] for ii in range(n_runs)]
            tmp = pool.starmap(func, args)
        err_list = [tmp[ii][0] for ii in range(n_runs)]
        var_list = [tmp[ii][1] for ii in range(n_runs)]
        ch_lists = [tmp[ii][2] for ii in range(n_runs)]
        ch_lists = [list(l) for l in zip(*ch_lists)]
        del tmp
        # err_list, var_list, ch_lists = func(17, cno_fold, is_array, truth_file, n_ch)

        # remove mean error of data points
        err_mean = sum(err_list) / n_runs
        err_df = pd.concat(err_list)  # - err_mean
        err_df2 = err_df - err_mean
        var_df = pd.concat(var_list)

        # Analytical
        res.loc[kk, "n"] = np.mean(var_df["P0"])
        res.loc[kk, "e"] = np.mean(var_df["P1"])
        res.loc[kk, "d"] = np.mean(var_df["P2"])
        res.loc[kk, "vn"] = np.mean(var_df["P3"])
        res.loc[kk, "ve"] = np.mean(var_df["P4"])
        res.loc[kk, "vd"] = np.mean(var_df["P5"])
        res.loc[kk, "r"] = np.mean(var_df["P6"])
        res.loc[kk, "p"] = np.mean(var_df["P7"])
        res.loc[kk, "y"] = np.mean(var_df["P8"])
        res.loc[kk, "cb"] = np.mean(var_df["P9"])
        res.loc[kk, "cd"] = np.mean(var_df["P10"])

        # Empirical
        res.loc[kk + 11, "n"] = np.var(err_df2["N"])
        res.loc[kk + 11, "e"] = np.var(err_df2["E"])
        res.loc[kk + 11, "d"] = np.var(err_df2["D"])
        res.loc[kk + 11, "vn"] = np.var(err_df2["vN"])
        res.loc[kk + 11, "ve"] = np.var(err_df2["vE"])
        res.loc[kk + 11, "vd"] = np.var(err_df2["vD"])
        res.loc[kk + 11, "r"] = np.var(err_df2["Roll"])
        res.loc[kk + 11, "p"] = np.var(err_df2["Pitch"])
        res.loc[kk + 11, "y"] = np.var(err_df2["Yaw"])
        res.loc[kk + 11, "cb"] = np.var(err_df2["Bias"])
        res.loc[kk + 11, "cd"] = np.var(err_df2["Drift"])

        # RMSE
        res.loc[kk + 22, "n"] = np.sqrt(np.mean(err_df["N"] ** 2))
        res.loc[kk + 22, "e"] = np.sqrt(np.mean(err_df["E"] ** 2))
        res.loc[kk + 22, "d"] = np.sqrt(np.mean(err_df["D"] ** 2))
        res.loc[kk + 22, "vn"] = np.sqrt(np.mean(err_df["vN"] ** 2))
        res.loc[kk + 22, "ve"] = np.sqrt(np.mean(err_df["vE"] ** 2))
        res.loc[kk + 22, "vd"] = np.sqrt(np.mean(err_df["vD"] ** 2))
        res.loc[kk + 22, "r"] = np.sqrt(np.mean(err_df["Roll"] ** 2))
        res.loc[kk + 22, "p"] = np.sqrt(np.mean(err_df["Pitch"] ** 2))
        res.loc[kk + 22, "y"] = np.sqrt(np.mean(err_df["Yaw"] ** 2))
        res.loc[kk + 22, "cb"] = np.sqrt(np.mean(err_df["Bias"] ** 2))
        res.loc[kk + 22, "cd"] = np.sqrt(np.mean(err_df["Drift"] ** 2))

        if MODEL == "Signal":
            res.loc[kk, "r"] *= RAD2DEG**2
            res.loc[kk, "p"] *= RAD2DEG**2
            res.loc[kk, "y"] *= RAD2DEG**2
            res.loc[kk, "cb"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk, "cd"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk + 11, "cb"] *= (1e9 / LIGHT_SPEED) ** 2
            res.loc[kk + 11, "cd"] *= (1e9 / LIGHT_SPEED) ** 2

        # Channels
        for i in range(n_ch):
            c_mean = sum(ch_lists[i]) / n_runs
            c = pd.concat(ch_lists[i])  # - c_mean
            c2 = c - c_mean
            ch_res[i].loc[kk, "dR"] = BETA**2 * DllVariance(
                10 ** (ch_res[i].loc[kk, "CNo"] / 10), 0.02
            )
            ch_res[i].loc[kk, "dRR"] = LAMBDA**2 * FllVariance(
                10 ** (ch_res[i].loc[kk, "CNo"] / 10), 0.02
            )
            a = 2 * PllVariance(10 ** (ch_res[i].loc[kk, "CNo"] / 10), 0.02) * RAD2DEG**2
            ch_res[i].loc[kk, "dP1"] = a
            ch_res[i].loc[kk, "dP2"] = a
            ch_res[i].loc[kk, "dP3"] = a
            ch_res[i].loc[kk + 11, "dR"] = np.var(c2["dR"])
            ch_res[i].loc[kk + 11, "dRR"] = np.var(c2["dRR"])
            ch_res[i].loc[kk + 11, "dP1"] = np.var(c2["dP1"]) * RAD2DEG**2
            ch_res[i].loc[kk + 11, "dP2"] = np.var(c2["dP2"]) * RAD2DEG**2
            ch_res[i].loc[kk + 11, "dP3"] = np.var(c2["dP3"]) * RAD2DEG**2
            ch_res[i].loc[kk + 22, "dR"] = np.sqrt(np.mean(c["dR"] ** 2))
            ch_res[i].loc[kk + 22, "dRR"] = np.sqrt(np.mean(c["dRR"] ** 2))
            ch_res[i].loc[kk + 22, "dP1"] = np.sqrt(np.mean(c["dP1"] ** 2)) * RAD2DEG
            ch_res[i].loc[kk + 22, "dP2"] = np.sqrt(np.mean(c["dP2"] ** 2)) * RAD2DEG
            ch_res[i].loc[kk + 22, "dP3"] = np.sqrt(np.mean(c["dP3"] ** 2)) * RAD2DEG

    # save to csv
    res_file = directory / "nav_results2.csv"
    res.to_csv(res_file)
    for i in range(n_ch):
        res_file = directory / f"channel_{i}_results2.csv"
        ch_res[i].to_csv(res_file)
    return res, ch_res


if __name__ == "__main__":
    # dir = Path(f"/mnt/f/Thesis-Data/{MODEL}-Sim/{SIM}-sim")
    dir = Path(f"/media/daniel/Sturdivant/Thesis-Data/{MODEL}-Sim/{SIM}-sim-vdfll")
    if SIM == "drone":
        res, ch_res = CalcMCStats(dir, False, 10)
    else:
        res, ch_res = CalcMCStats(dir, False, 11)

    print(res)
