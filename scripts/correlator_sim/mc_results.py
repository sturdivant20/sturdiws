import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6 import QtWidgets

sys.path.append("scripts")
from utils.parsers import ParseCorrelatorSimLogs
from utils.plotters import MyWindow, MatplotlibWidget


def ProcessResults(directory: Path, is_array: bool = True):
    res_file = directory / "mc_results.csv"
    if res_file.is_file():
        res = pd.read_csv(res_file)

    else:
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
        res["J2S"] = np.arange(43.5, 21.5, -2.0)

        # loop through each j2s
        cno_folders = sorted([d for d in directory.iterdir()])
        for kk, cno_folder in enumerate(cno_folders):
            # calculate variance over all samples of 100 runs
            err_list = []
            var_list = []
            for ii in range(100):
                mc_folder = cno_folder / f"Run{ii}"
                nav, err, _ = ParseCorrelatorSimLogs(mc_folder, is_array)
                err_list.append(err.loc[err["t"] > 30.0])
                var_list.append(nav.iloc[-1000:])
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

        # save to csv
        res.to_csv(res_file)
    return res


if __name__ == "__main__":
    res = ProcessResults(Path("/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/drone-sim"))
    # res = ProcessResults(Path("/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/ground-sim"))
    print(res)

    COLORS = ["#100c08", "#a52a2a", "#a2e3b8", "#324ab2", "#c5961d", "#454d32", "#c8c8c8"]
    sns.set_theme(
        font="Times New Roman",
        context="paper",
        style="ticks",
        palette=sns.color_palette(COLORS),
        rc={
            "axes.grid": True,
            "grid.linestyle": ":",
            "lines.linewidth": 2,
        },
        font_scale=1.5,
    )

    # create window
    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow()

    # position variance
    myp = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=res["KFn"], marker=">", label="KF", ax=myp.ax[0])
    sns.lineplot(x=res["CNo"], y=res["MCn"], marker="o", label="MC", ax=myp.ax[0])
    myp.ax[0].set(ylabel="North [m$^2$]", yscale="log")
    myp.ax[0].minorticks_on()
    myp.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFe"], marker=">", ax=myp.ax[1])
    sns.lineplot(x=res["CNo"], y=res["MCe"], marker="o", ax=myp.ax[1])
    myp.ax[1].set(ylabel="East [m$^2$]", yscale="log")
    myp.ax[1].minorticks_on()
    myp.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFd"], marker=">", ax=myp.ax[2])
    sns.lineplot(x=res["CNo"], y=res["MCd"], marker="o", ax=myp.ax[2])
    myp.ax[2].set(
        ylabel="Down [m$^2$]",
        xlabel="C/N$_0$ [dB-Hz]",
        xticks=range(20, 42, 2),
        yscale="log",
    )
    myp.ax[1].minorticks_on()
    myp.ax[1].grid(which="minor", alpha=0.4)
    myp.f.tight_layout()
    win.NewTab(myp, "Position")

    # velocity variance
    myv = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=res["KFvn"], marker=">", label="KF", ax=myv.ax[0])
    sns.lineplot(x=res["CNo"], y=res["MCvn"], marker="o", label="MC", ax=myv.ax[0])
    myv.ax[0].set(ylabel="North [(m/s)$^2$]", yscale="log")
    myv.ax[0].minorticks_on()
    myv.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFve"], marker=">", ax=myv.ax[1])
    sns.lineplot(x=res["CNo"], y=res["MCve"], marker="o", ax=myv.ax[1])
    myv.ax[1].set(ylabel="East [(m/s)$^2$]", yscale="log")
    myv.ax[1].minorticks_on()
    myv.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFvd"], marker=">", ax=myv.ax[2])
    sns.lineplot(x=res["CNo"], y=res["MCvd"], marker="o", ax=myv.ax[2])
    myv.ax[2].set(
        ylabel="Down [(m/s)$^2$]",
        xlabel="C/N$_0$ [dB-Hz]",
        xticks=range(20, 42, 2),
        yscale="log",
    )
    myv.ax[2].minorticks_on()
    myv.ax[2].grid(which="minor", alpha=0.4)
    myv.f.tight_layout()
    win.NewTab(myv, "Velocity")

    # attitude variance
    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=res["KFr"], marker=">", label="KF", ax=mya.ax[0])
    sns.lineplot(x=res["CNo"], y=res["MCr"], marker="o", label="MC", ax=mya.ax[0])
    mya.ax[0].set(ylabel="Roll [deg$^2$]", yscale="log")
    mya.ax[0].minorticks_on()
    mya.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFp"], marker=">", ax=mya.ax[1])
    sns.lineplot(x=res["CNo"], y=res["MCp"], marker="o", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [deg$^2$]", yscale="log")
    mya.ax[1].minorticks_on()
    mya.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFy"], marker=">", ax=mya.ax[2])
    sns.lineplot(x=res["CNo"], y=res["MCy"], marker="o", ax=mya.ax[2])
    mya.ax[2].set(
        ylabel="Yaw [deg$^2$]",
        xlabel="C/N$_0$ [dB-Hz]",
        xticks=range(20, 42, 2),
        yscale="log",
    )
    mya.ax[2].minorticks_on()
    mya.ax[2].grid(which="minor", alpha=0.4)
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")

    # clock variance
    myc = MatplotlibWidget(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=res["KFcb"], marker=">", label="KF", ax=myc.ax[0])
    sns.lineplot(x=res["CNo"], y=res["MCcb"], marker="o", label="MC", ax=myc.ax[0])
    myc.ax[0].set(ylabel="Bias [ns$^2$]", yscale="log")
    myc.ax[0].minorticks_on()
    myc.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=res["CNo"], y=res["KFcb"], marker=">", ax=myc.ax[1])
    sns.lineplot(x=res["CNo"], y=res["MCcb"], marker="o", ax=myc.ax[1])
    myc.ax[1].set(
        ylabel="Drift [(ns/s)$^2$]",
        xlabel="C/N$_0$ [dB-Hz]",
        xticks=range(20, 42, 2),
        yscale="log",
    )
    myc.ax[1].minorticks_on()
    myc.ax[1].grid(which="minor", alpha=0.4)
    myc.f.tight_layout()
    win.NewTab(myc, "Clock")

    # open plots
    win.show()
    sys.exit(app.exec())
