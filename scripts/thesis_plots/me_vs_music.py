"""
Show that my attitude is better that MUSIC+Wahba
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from navtools._navtools_core.attitude import quat2euler

sys.path.append("scripts")
from utils.plotters import MyWindow, MatplotlibWidget
from utils.parsers import ParseCorrelatorSimLogs, ParseNavSimStates

if __name__ == "__main__":
    DATASET = "drone"  # "ground"
    LEN = 10  # 11
    SAVE = True

    dir1 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{DATASET}-sim")
    dir2 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{DATASET}-sim-music")
    mc = pd.concat(
        [
            pd.read_csv(dir1 / "nav_results.csv", index_col=0),
            pd.read_csv(dir2 / "nav_results.csv", index_col=0),
        ]
    )
    mc["Sim"] = "Proposed"
    mc.iloc[33:, -1] = "MUSIC+Wahba"
    mc_var = mc[mc["Type"] != "RMSE"]
    mc_rmse = mc[mc["Type"] == "RMSE"]
    mc_var.iloc[:, 3:-1] = 10 * np.log10(mc_var.iloc[:, 3:-1])

    COLORS = ["#100c08", "#a52a2a", "#a2e3b8", "#324ab2", "#c5961d", "#454d32", "#c8c8c8"]
    sns.set_theme(
        font="Times New Roman",
        context="paper",
        style="ticks",
        palette=sns.color_palette(COLORS),
        rc={
            "axes.grid": True,
            "grid.linestyle": ":",
            "lines.linewidth": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.default": "it",
        },
        font_scale=1.5,
    )

    # !--- Attitude Variance ----------
    myavar = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="r",
        hue="Sim",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myavar.ax[0],
    )
    l.legend().set_title("")
    sns.move_legend(myavar.ax[0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    myavar.ax[0].set(ylabel=r"$\sigma^2_{Roll}$ [dB-$\circ^2$]")  # , yscale="log")
    myavar.ax[0].minorticks_on()
    myavar.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="p",
        hue="Sim",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myavar.ax[1],
    )
    l.legend_.remove()
    myavar.ax[1].set(ylabel=r"$\sigma^2_{Pitch}$ [dB-$\circ^2$]")  # , yscale="log")
    myavar.ax[1].minorticks_on()
    myavar.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="y",
        hue="Sim",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myavar.ax[2],
    )
    l.legend_.remove()
    myavar.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"$\sigma^2_{Yaw}$ [dB-$\circ^2$]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    myavar.ax[2].minorticks_on()
    myavar.ax[2].grid(which="minor", alpha=0.4)
    myavar.f.tight_layout()

    # !--- Attitude RMSE -----------
    myarmse = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="r",
        hue="Sim",
        style="Type",
        markers=["o"],
        dashes=[(3.5, 1.5)],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[0],
    )
    l.legend().set_title("")
    sns.move_legend(myarmse.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    myarmse.ax[0].set(ylabel=r"$RMSE_{Roll}$ [$\circ$]")  # , yscale="log")
    myarmse.ax[0].minorticks_on()
    myarmse.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="p",
        hue="Sim",
        style="Type",
        markers=["o"],
        dashes=[(3.5, 1.5)],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[1],
    )
    l.legend_.remove()
    myarmse.ax[1].set(ylabel=r"$RMSE_{Pitch}$ [$\circ$]")  # , yscale="log")
    myarmse.ax[1].minorticks_on()
    myarmse.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="y",
        hue="Sim",
        style="Type",
        markers=["o"],
        dashes=[(3.5, 1.5)],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[2],
    )
    l.legend_.remove()
    myarmse.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"$RMSE_{Yaw}$ [$\circ$]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    myarmse.ax[2].minorticks_on()
    myarmse.ax[2].grid(which="minor", alpha=0.4)
    myarmse.f.tight_layout()

    #! --- Single Run Estimates ---
    nav1, err1, _ = ParseCorrelatorSimLogs(
        "/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/drone-sim/CNo_30_dB/Run1", False
    )
    rpy = np.zeros((len(nav1), 3), order="F")
    for ii in range(len(nav1)):
        q = np.array(
            [nav1.loc[ii, "qw"], nav1.loc[ii, "qx"], nav1.loc[ii, "qy"], nav1.loc[ii, "qz"]],
            order="F",
        )
        rpy[ii, :] = np.rad2deg(quat2euler(q, True))
    nav1["Sim"] = "Proposed"
    nav1["Roll"] = rpy[:, 0]
    nav1["Pitch"] = rpy[:, 1]
    nav1["Yaw"] = rpy[:, 2]
    err1["Sim"] = "Proposed"
    nav2, err2, _ = ParseCorrelatorSimLogs(
        "/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/drone-sim-music/CNo_30_dB/Run2", False
    )
    rpy = np.zeros((len(nav2), 3), order="F")
    for ii in range(len(nav2)):
        q = np.array(
            [nav2.loc[ii, "qw"], nav2.loc[ii, "qx"], nav2.loc[ii, "qy"], nav2.loc[ii, "qz"]],
            order="F",
        )
        rpy[ii, :] = np.rad2deg(quat2euler(q, True))
    nav2["Sim"] = "MUSIC+Wahba"
    nav2["Roll"] = rpy[:, 0]
    nav2["Pitch"] = rpy[:, 1]
    nav2["Yaw"] = rpy[:, 2]
    err2["Sim"] = "MUSIC+Wahba"
    truth = ParseNavSimStates("data/drone_sim.bin")
    truth["Sim"] = "Truth"
    truth.rename(columns={"r": "Roll", "p": "Pitch", "y": "Yaw"}, inplace=True)
    truth["t"] = truth["t"] / 1000.0
    att = pd.concat(
        [
            nav1.loc[:, ["t", "Sim", "Roll", "Pitch", "Yaw"]],
            nav2.loc[:, ["t", "Sim", "Roll", "Pitch", "Yaw"]],
            truth.loc[:, ["t", "Sim", "Roll", "Pitch", "Yaw"]],
        ]
    )
    t_map = np.round(nav2["t"], 2)
    err2["Roll"] = (
        truth.loc[truth[np.in1d(truth["t"], t_map)].index, "Roll"].values - nav2["Roll"].values
    )
    err2["Pitch"] = (
        truth.loc[truth[np.in1d(truth["t"], t_map)].index, "Pitch"].values - nav2["Pitch"].values
    )
    # err2["Yaw"] = (
    #     truth.loc[truth[np.in1d(truth["t"], t_map)].index, "Yaw"].values - nav2["Yaw"].values
    # )
    # err2.loc[err2["Yaw"] > 180, "Yaw"] = err2.loc[err2["Yaw"] > 180, "Yaw"].values - 360
    # err2.loc[err2["Yaw"] < -180, "Yaw"] = err2.loc[err2["Yaw"] < -180, "Yaw"].values + 360
    att_err = pd.concat(
        [
            err1.loc[:, ["t", "Sim", "Roll", "Pitch", "Yaw"]],
            err2.loc[:, ["t", "Sim", "Roll", "Pitch", "Yaw"]],
        ]
    )

    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=att,
        x="t",
        y="Roll",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=mya.ax[0],
    )
    l.legend().set_title("")
    sns.move_legend(mya.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
    mya.ax[0].set(ylabel=r"Roll [$\circ$]")  # , yscale="log")
    mya.ax[0].minorticks_on()
    mya.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=att,
        x="t",
        y="Pitch",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=mya.ax[1],
    )
    l.legend_.remove()
    mya.ax[1].set(ylabel=r"Pitch [$\circ$]")  # , yscale="log")
    mya.ax[1].minorticks_on()
    mya.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=att,
        x="t",
        y="Yaw",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=mya.ax[2],
    )
    l.legend_.remove()
    mya.ax[2].set(xlabel=r"Time [s]", ylabel=r"Yaw [$\circ$]")  # , yscale="log")
    mya.ax[2].minorticks_on()
    mya.ax[2].grid(which="minor", alpha=0.4)
    mya.f.tight_layout()

    #! --- Single Rune Error ---
    myaerr = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=att_err,
        x="t",
        y="Roll",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=myaerr.ax[0],
    )
    l.legend().set_title("")
    sns.move_legend(myaerr.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    myaerr.ax[0].set(ylabel=r"Roll [$\circ$]")  # , yscale="log")
    myaerr.ax[0].minorticks_on()
    myaerr.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=att_err,
        x="t",
        y="Pitch",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=myaerr.ax[1],
    )
    l.legend_.remove()
    myaerr.ax[1].set(ylabel=r"Pitch [$\circ$]")  # , yscale="log")
    myaerr.ax[1].minorticks_on()
    myaerr.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=att_err,
        x="t",
        y="Yaw",
        hue="Sim",
        errorbar=None,
        markersize=8,
        ax=myaerr.ax[2],
    )
    l.legend_.remove()
    myaerr.ax[2].set(xlabel=r"Time [s]", ylabel=r"Yaw [$\circ$]")  # , yscale="log")
    myaerr.ax[2].minorticks_on()
    myaerr.ax[2].grid(which="minor", alpha=0.4)
    myaerr.f.tight_layout()

    # save plots
    if SAVE:
        outdir = Path("/media/daniel/Sturdivant/Thesis-Data/MC-Results-Plots/")
        outdir.mkdir(parents=True, exist_ok=True)
        myavar.f.savefig(
            outdir / f"me_vs_music_{DATASET}_sim_attitude_var.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        myarmse.f.savefig(
            outdir / f"me_vs_music_{DATASET}_sim_attitude_rmse.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        mya.f.savefig(
            outdir / f"me_vs_music_{DATASET}_sim_attitude_estimate_30_dBHz.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        myaerr.f.savefig(
            outdir / f"me_vs_music_{DATASET}_sim_attitude_error_30_dBHz.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()
