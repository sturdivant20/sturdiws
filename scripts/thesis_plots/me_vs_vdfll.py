"""
Show that my positioning is better than a regular VDFLL
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sys.path.append("scripts")
from utils.plotters import MyWindow, MatplotlibWidget

if __name__ == "__main__":
    DATASET = "drone"  # "ground"
    LEN = 10  # 11
    SAVE = True

    dir1 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{DATASET}-sim")
    dir2 = Path(f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{DATASET}-sim-vdfll")
    mc = pd.concat(
        [
            pd.read_csv(dir1 / "nav_results.csv", index_col=0),
            pd.read_csv(dir2 / "nav_results.csv", index_col=0),
        ]
    )
    mc["Sim"] = "Proposed"
    mc.iloc[33:, -1] = "VDFLL"
    mc_var = mc[mc["Type"] != "RMSE"]
    mc_rmse = mc[mc["Type"] == "RMSE"]
    mc_var.iloc[:, 3:-1] = 10 * np.log10(mc_var.iloc[:, 3:-1])

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

    # !--- Position Variance ----------
    mypvar = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="n",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=mypvar.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(mypvar.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    mypvar.ax[0].set(ylabel=r"$\sigma^2_{N}$ [dB-m$^2$]")  # , yscale="log")
    mypvar.ax[0].minorticks_on()
    mypvar.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="e",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=mypvar.ax[1],
    )
    l.legend_.remove()
    mypvar.ax[1].set(ylabel=r"$\sigma^2_{E}$ [dB-m$^2$]")  # , yscale="log")
    mypvar.ax[1].minorticks_on()
    mypvar.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="d",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=mypvar.ax[2],
    )
    l.legend_.remove()
    mypvar.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"$\sigma^2_{D}$ [dB-m$^2$]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    mypvar.ax[2].minorticks_on()
    mypvar.ax[2].grid(which="minor", alpha=0.4)
    mypvar.f.tight_layout()

    # !--- Position RMSE -----------
    myprmse = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="n",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myprmse.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myprmse.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    myprmse.ax[0].set(ylabel=r"RMSE N [m]")  # , yscale="log")
    myprmse.ax[0].minorticks_on()
    myprmse.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="e",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myprmse.ax[1],
    )
    l.legend_.remove()
    myprmse.ax[1].set(ylabel=r"RMSE E [m]")  # , yscale="log")
    myprmse.ax[1].minorticks_on()
    myprmse.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="d",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myprmse.ax[2],
    )
    l.legend_.remove()
    myprmse.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"RMSE D [m]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    myprmse.ax[2].minorticks_on()
    myprmse.ax[2].grid(which="minor", alpha=0.4)
    myprmse.f.tight_layout()

    # !--- Velocity Variance ----------
    myvvar = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="vn",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvvar.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myvvar.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    myvvar.ax[0].set(ylabel=r"$\sigma^2_{\dot{N}}$ [dB-(m/s)$^2$]")  # , yscale="log")
    myvvar.ax[0].minorticks_on()
    myvvar.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="ve",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvvar.ax[1],
    )
    l.legend_.remove()
    myvvar.ax[1].set(ylabel=r"$\sigma^2_{\dot{E}}$ [dB-(m/s)$^2$]")  # , yscale="log")
    myvvar.ax[1].minorticks_on()
    myvvar.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_var,
        x="CNo",
        y="vd",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvvar.ax[2],
    )
    l.legend_.remove()
    myvvar.ax[2].set(
        xlabel=r"C/No [dB-Hz]",
        ylabel=r"$\sigma^2_{\dot{D}}$ [dB-(m/s)$^2$]",
        xticks=range(20, 42, 2),
    )  # , yscale="log")
    myvvar.ax[2].minorticks_on()
    myvvar.ax[2].grid(which="minor", alpha=0.4)
    myvvar.f.tight_layout()

    # !--- Velocity RMSE ----------
    myvrmse = MatplotlibWidget(nrows=3, ncols=1, figsize=(6, 6), sharex=True)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="vn",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvrmse.ax[0],
    )
    handles = l.legend().legend_handles
    for i in range(len(handles)):
        if i > 3:
            handles[i].set_color(COLORS[6])
    sns.move_legend(myvrmse.ax[0], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    myvrmse.ax[0].set(ylabel=r"RMSE $\dot{N}$ [m/s]")  # , yscale="log")
    myvrmse.ax[0].minorticks_on()
    myvrmse.ax[0].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="ve",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvrmse.ax[1],
    )
    l.legend_.remove()
    myvrmse.ax[1].set(ylabel=r"RMSE $\dot{E}$ [m/s]")  # , yscale="log")
    myvrmse.ax[1].minorticks_on()
    myvrmse.ax[1].grid(which="minor", alpha=0.4)
    l = sns.lineplot(
        data=mc_rmse,
        x="CNo",
        y="vd",
        hue="Sim",
        marker=">",
        errorbar=None,
        markersize=8,
        ax=myvrmse.ax[2],
    )
    l.legend_.remove()
    myvrmse.ax[2].set(
        xlabel=r"C/No [dB-Hz]", ylabel=r"RMSE $\dot{D}$ [m/s]", xticks=range(20, 42, 2)
    )  # , yscale="log")
    myvrmse.ax[2].minorticks_on()
    myvrmse.ax[2].grid(which="minor", alpha=0.4)
    myvrmse.f.tight_layout()

    # save plots
    if SAVE:
        outdir = Path("/media/daniel/Sturdivant/Thesis-Data/MC-Results-Plots/")
        outdir.mkdir(parents=True, exist_ok=True)
        mypvar.f.savefig(
            outdir / f"me_vs_vdfll_{DATASET}_sim_position_var.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        myvvar.f.savefig(
            outdir / f"me_vs_vdfll_{DATASET}_sim_velocity_var.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        myprmse.f.savefig(
            outdir / f"me_vs_vdfll_{DATASET}_sim_position_rmse.svg",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        myvrmse.f.savefig(
            outdir / f"me_vs_vdfll_{DATASET}_sim_velocity_rmse.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )

    # create window
    # app = QtWidgets.QApplication(sys.argv)
    # win = MyWindow()
    # win.NewTab(myavar, "Attitude Variance")
    # # win.NewTab(myarmse, "Attitude RMSE")
    # # win.NewTab(myspvar, "Spatial Phase Variance")
    # # win.NewTab(mysprmse, "Spatial Phase RMSE")
    # win.show()
    # sys.exit(app.exec())
    plt.show()
