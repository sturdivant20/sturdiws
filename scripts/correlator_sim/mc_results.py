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
        kk = 0
        res["CNo"] = np.array([40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20])
        res["J2S"] = np.array([22, 24.7, 27, 29.2, 31.3, 33.4, 35.5, 37.5, 39.5, 41.5, 43.5])

        # loop through each j2s
        j2s_folders = sorted([d for d in directory.iterdir()])
        for j2s_folder in j2s_folders:
            # calculate variance over all samples of 100 runs
            err_list = []
            var_list = []
            for ii in range(100):
                mc_folder = j2s_folder / f"{ii}"
                _, err, var, _ = ParseCorrelatorSimLogs(mc_folder, is_array)
                err_list.append(err)
                var_list.append(var)
            err_df = pd.concat(err_list)
            var_df = pd.concat(var_list)
            t_end = var_list[0].iloc[-1, 0]

            res.loc[kk, "MCn"] = np.var(err_df["lat"])
            res.loc[kk, "MCe"] = np.var(err_df["lon"])
            res.loc[kk, "MCd"] = np.var(err_df["h"])
            res.loc[kk, "MCvn"] = np.var(err_df["vn"])
            res.loc[kk, "MCve"] = np.var(err_df["ve"])
            res.loc[kk, "MCvd"] = np.var(err_df["vd"])
            res.loc[kk, "MCr"] = np.var(err_df["r"])
            res.loc[kk, "MCp"] = np.var(err_df["p"])
            res.loc[kk, "MCy"] = np.var(err_df["y"])
            res.loc[kk, "MCcb"] = np.var(err_df["cb"])
            res.loc[kk, "MCcd"] = np.var(err_df["cd"])

            res.loc[kk, "KFn"] = np.mean(var_df.loc[var_df["t"] == t_end, "lat"])
            res.loc[kk, "KFe"] = np.mean(var_df.loc[var_df["t"] == t_end, "lon"])
            res.loc[kk, "KFd"] = np.mean(var_df.loc[var_df["t"] == t_end, "h"])
            res.loc[kk, "KFvn"] = np.mean(var_df.loc[var_df["t"] == t_end, "vn"])
            res.loc[kk, "KFve"] = np.mean(var_df.loc[var_df["t"] == t_end, "ve"])
            res.loc[kk, "KFvd"] = np.mean(var_df.loc[var_df["t"] == t_end, "vd"])
            res.loc[kk, "KFr"] = np.mean(var_df.loc[var_df["t"] == t_end, "r"])
            res.loc[kk, "KFp"] = np.mean(var_df.loc[var_df["t"] == t_end, "p"])
            res.loc[kk, "KFy"] = np.mean(var_df.loc[var_df["t"] == t_end, "y"])
            res.loc[kk, "KFcb"] = np.mean(var_df.loc[var_df["t"] == t_end, "cb"])
            res.loc[kk, "KFcd"] = np.mean(var_df.loc[var_df["t"] == t_end, "cd"])

            kk += 1

        # save to csv
        res.to_csv(res_file)
    return res


if __name__ == "__main__":
    res = ProcessResults(Path("/media/daniel/Sturdivant/Thesis-Results/Correlator-Sim/drone-sim"))
    # res = ProcessResults(Path("/media/daniel/Sturdivant/Thesis-Results/Correlator-Sim/ground-sim"))

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
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFn"]), marker=">", label="KF", ax=myp.ax[0])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCn"]), marker="o", label="MC", ax=myp.ax[0])
    myp.ax[0].set(ylabel="North [dB-m$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFe"]), marker=">", ax=myp.ax[1])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCe"]), marker="o", ax=myp.ax[1])
    myp.ax[1].set(ylabel="East [dB-m$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFd"]), marker=">", ax=myp.ax[2])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCd"]), marker="o", ax=myp.ax[2])
    myp.ax[2].set(ylabel="Down [dB-m$^2$]", xlabel="C/N$_0$ [dB-Hz]", xticks=range(20, 42, 2))
    myp.f.tight_layout()
    win.NewTab(myp, "Position")

    # velocity variance
    myv = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFvn"]), marker=">", label="KF", ax=myv.ax[0])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCvn"]), marker="o", label="MC", ax=myv.ax[0])
    myv.ax[0].set(ylabel="North [dB-(m/s)$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFve"]), marker=">", ax=myv.ax[1])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCve"]), marker="o", ax=myv.ax[1])
    myv.ax[1].set(ylabel="East [dB-(m/s)$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFve"]), marker=">", ax=myv.ax[2])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCve"]), marker="o", ax=myv.ax[2])
    myv.ax[2].set(ylabel="Down [dB-(m/s)$^2$]", xlabel="C/N$_0$ [dB-Hz]", xticks=range(20, 42, 2))
    myv.f.tight_layout()
    win.NewTab(myv, "Velocity")

    # attitude variance
    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFr"]), marker=">", label="KF", ax=mya.ax[0])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCr"]), marker="o", label="MC", ax=mya.ax[0])
    mya.ax[0].set(ylabel="Roll [dB-deg$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFp"]), marker=">", ax=mya.ax[1])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCp"]), marker="o", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [dB-deg$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFy"]), marker=">", ax=mya.ax[2])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCy"]), marker="o", ax=mya.ax[2])
    mya.ax[2].set(ylabel="Yaw [dB-deg$^2$]", xlabel="C/N$_0$ [dB-Hz]", xticks=range(20, 42, 2))
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")

    # attitude variance
    myc = MatplotlibWidget(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFcd"]), marker=">", label="KF", ax=myc.ax[0])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCcd"]), marker="o", label="MC", ax=myc.ax[0])
    myc.ax[0].set(ylabel="Bias [dB-ns$^2$]")
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["KFcd"]), marker=">", ax=myc.ax[1])
    sns.lineplot(x=res["CNo"], y=10 * np.log10(res["MCcd"]), marker="o", ax=myc.ax[1])
    myc.ax[1].set(ylabel="Drift [dB-(ns/s)$^2$]", xlabel="C/N$_0$ [dB-Hz]", xticks=range(20, 42, 2))
    myc.f.tight_layout()
    win.NewTab(myc, "Clock")

    # open plots
    win.show()
    sys.exit(app.exec())
