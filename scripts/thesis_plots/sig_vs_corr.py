"""
Show that my attitude estimates are consistent across both correlator and signal simulations
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sys.path.append("scripts")
from PyQt6 import QtWidgets
from utils.plotters import MyWindow, MatplotlibWidget

if __name__ == "__main__":
    DATASET = "ground"  # "drone"
    LEN = 11  # 10
    SAVE = True

    # dir1 = Path(f"/mnt/f/Thesis-Data/Signal-Sim/{DATASET}-sim")
    # dir2 = Path(f"/mnt/f/Thesis-Data/Correlator-Sim/{DATASET}-sim")
    dir1 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/{DATASET}-sim")
    dir2 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{DATASET}-sim")
    mc = pd.concat(
        [
            pd.read_csv(dir1 / "nav_results.csv", index_col=0),
            pd.read_csv(dir2 / "nav_results.csv", index_col=0),
        ]
    )
    mc_var = mc[mc["Type"] != "RMSE"]
    mc_rmse = mc[mc["Type"] == "RMSE"]
    mc_var.iloc[:, 3:] = 10 * np.log10(mc_var.iloc[:, 3:])
    # mc_rmse.iloc[:, 3:] = 20 * np.log10(mc_rmse.iloc[:, 3:])

    for ii in range(LEN):
        if ii == 0:
            ch = pd.concat(
                [
                    pd.read_csv(dir1 / f"channel_{ii}_results.csv", index_col=0),
                    pd.read_csv(dir2 / f"channel_{ii}_results.csv", index_col=0),
                ]
            )
        else:
            tmp = pd.concat(
                [
                    pd.read_csv(dir1 / f"channel_{ii}_results.csv", index_col=0),
                    pd.read_csv(dir2 / f"channel_{ii}_results.csv", index_col=0),
                ]
            )
            ch.iloc[:, 3:] += tmp.iloc[:, 3:]
    ch.iloc[:, 3:] /= LEN
    ch_var = ch[ch["Type"] != "RMSE"]
    ch_rmse = ch[ch["Type"] == "RMSE"]
    ch_var.iloc[:, 3:] = 10 * np.log10(ch_var.iloc[:, 3:])
    # ch_rmse.iloc[:, 3:] = 20 * np.log10(ch_rmse.iloc[:, 3:])
    # print(ch_rmse)

    COLORS = ["#100c08", "#a52a2a", "#324ab2", "#c5961d", "#454d32", "#c8c8c8", "#a2e3b8"]
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

    # create window
    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow(f"{DATASET} sim monte carlo signal vs correlator models")

    # !--- Attitude Variance ----------
    myavar = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="r",
        hue="Model",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myavar.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myavar.ax[0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    myavar.ax[0].set(ylabel=r"$\sigma^2_{Roll}$ [dB-$\circ^2$]")  # , yscale="log")
    myavar.ax[0].minorticks_on()
    myavar.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="p",
        hue="Model",
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
        hue="Model",
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
    win.NewTab(myavar, "Attitude Variance")

    # !--- Attitude RMSE -----------
    myarmse = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="r",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myarmse.ax[0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    myarmse.ax[0].set(ylabel=r"RMSE Roll [$\circ$]")  # , yscale="log")
    myarmse.ax[0].minorticks_on()
    myarmse.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="p",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[1],
    )
    l.legend_.remove()
    myarmse.ax[1].set(ylabel=r"RMSE Pitch [$\circ$]")  # , yscale="log")
    myarmse.ax[1].minorticks_on()
    myarmse.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="y",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=myarmse.ax[2],
    )
    l.legend_.remove()
    myarmse.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"RMSE Yaw [$\circ$]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    myarmse.ax[2].minorticks_on()
    myarmse.ax[2].grid(which="minor", alpha=0.4)
    myarmse.f.tight_layout()
    win.NewTab(myarmse, "Attitude RMSE")

    # !--- Spatial Phase Variance ----------
    myspvar = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=ch_var,
        x="CNo",
        y="dP1",
        hue="Model",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myspvar.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myspvar.ax[0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    myspvar.ax[0].set(ylabel=r"$\sigma^2_{\Delta\phi_1}$ [dB-$\circ^2$]")  # , yscale="log")
    myspvar.ax[0].minorticks_on()
    myspvar.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=ch_var,
        x="CNo",
        y="dP2",
        hue="Model",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myspvar.ax[1],
    )
    l.legend_.remove()
    myspvar.ax[1].set(ylabel=r"$\sigma^2_{\Delta\phi_2}$ [dB-$\circ^2$]")  # , yscale="log")
    myspvar.ax[1].minorticks_on()
    myspvar.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=ch_var,
        x="CNo",
        y="dP3",
        hue="Model",
        style="Type",
        markers=[">", "o"],
        errorbar=None,
        markersize=8,
        ax=myspvar.ax[2],
    )
    l.legend_.remove()
    myspvar.ax[2].set(
        xlabel=r"C/No [dB-Hz]",
        ylabel=r"$\sigma^2_{\Delta\phi_3}$ [dB-$\circ^2$]",
        xticks=range(20, 42, 2),
    )  # , yscale="log")
    myspvar.ax[2].minorticks_on()
    myspvar.ax[2].grid(which="minor", alpha=0.4)
    myspvar.f.tight_layout()
    win.NewTab(myspvar, "Spatial Phase Variance")

    # !--- Spatial Phase RMSE ----------
    mysprmse = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=ch_rmse,
        x="CNo",
        y="dP1",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=mysprmse.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(mysprmse.ax[0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    mysprmse.ax[0].set(ylabel=r"RMSE $\Delta\phi_1$ [$\circ$]")  # , yscale="log")
    mysprmse.ax[0].minorticks_on()
    mysprmse.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=ch_rmse,
        x="CNo",
        y="dP2",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=mysprmse.ax[1],
    )
    l.legend_.remove()
    mysprmse.ax[1].set(ylabel=r"RMSE $\Delta\phi_2$  [$\circ$]")  # , yscale="log")
    mysprmse.ax[1].minorticks_on()
    mysprmse.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=ch_rmse,
        x="CNo",
        y="dP3",
        hue="Model",
        style="Type",
        markers=[">"],
        errorbar=None,
        markersize=8,
        ax=mysprmse.ax[2],
    )
    l.legend_.remove()
    mysprmse.ax[2].set(
        xlabel=r"C/No [dB-Hz]",
        ylabel=r"RMSE $\Delta\phi_3$  [$\circ$]",
        xticks=range(20, 42, 2),
        # yscale="log",
    )
    mysprmse.ax[2].minorticks_on()
    mysprmse.ax[2].grid(which="minor", alpha=0.4)
    mysprmse.f.tight_layout()
    win.NewTab(mysprmse, "Spatial Phase RMSE")

    # save plots
    if SAVE:
        # outdir = Path("/mnt/f/Thesis-Data/MC-Results-Plots/")
        outdir = Path("/media/daniel/Sturdivant/Thesis-Data/MC-Results-Plots/")
        outdir.mkdir(parents=True, exist_ok=True)
        myavar.f.savefig(
            outdir / f"sig_vs_corr_{DATASET}_sim_attitude_var.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        myspvar.f.savefig(
            outdir / f"sig_vs_corr_{DATASET}_sim_spatial_phase_var.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        myarmse.f.savefig(
            outdir / f"sig_vs_corr_{DATASET}_sim_attitude_rmse.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        mysprmse.f.savefig(
            outdir / f"sig_vs_corr_{DATASET}_sim_spatial_phase_rmse.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )

    # win.show()
    # sys.exit(app.exec())
    plt.show()
