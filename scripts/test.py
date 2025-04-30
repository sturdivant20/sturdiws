import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6 import QtWidgets
from utils.plotters import MyWindow, MatplotlibWidget

if __name__ == "__main__":
    s = "drone"  # "ground"
    sig = pd.read_csv(f"/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/{s}-sim/mc_results.csv")
    corr = pd.read_csv(
        f"/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/{s}-sim/mc_results.csv"
    )

    COLORS = ["#100c08", "#a52a2a", "#324ab2", "#c5961d", "#454d32", "#c8c8c8", "#a2e3b8"]
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
    sns.lineplot(x=corr["CNo"], y=corr["KFn"], marker=">", label="Correlator KF", ax=myp.ax[0])
    sns.lineplot(x=corr["CNo"], y=corr["MCn"], marker="o", label="Correlator MC", ax=myp.ax[0])
    sns.lineplot(
        x=sig["CNo"],
        y=sig["KFn"],
        linestyle="--",
        marker=">",
        label="Signal KF",
        ax=myp.ax[0],
    )
    sns.lineplot(
        x=sig["CNo"], y=sig["MCn"], linestyle="--", marker="o", label="Signal MC", ax=myp.ax[0]
    )
    myp.ax[0].set(ylabel="North [m$^2$]", yscale="log")
    myp.ax[0].minorticks_on()
    myp.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFe"], marker=">", ax=myp.ax[1])
    sns.lineplot(x=corr["CNo"], y=corr["MCe"], marker="o", ax=myp.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["KFe"], linestyle="--", marker=">", ax=myp.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["MCe"], linestyle="--", marker="o", ax=myp.ax[1])
    myp.ax[1].set(ylabel="East [m$^2$]", yscale="log")
    myp.ax[1].minorticks_on()
    myp.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFd"], marker=">", ax=myp.ax[2])
    sns.lineplot(x=corr["CNo"], y=corr["MCd"], marker="o", ax=myp.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["KFd"], linestyle="--", marker=">", ax=myp.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["MCd"], linestyle="--", marker="o", ax=myp.ax[2])
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
    sns.lineplot(x=corr["CNo"], y=corr["KFvn"], marker=">", label="Correlator KF", ax=myv.ax[0])
    sns.lineplot(x=corr["CNo"], y=corr["MCvn"], marker="o", label="Correlator MC", ax=myv.ax[0])
    sns.lineplot(
        x=sig["CNo"],
        y=sig["KFvn"],
        linestyle="--",
        marker=">",
        label="Signal KF",
        ax=myv.ax[0],
    )
    sns.lineplot(
        x=sig["CNo"], y=sig["MCvn"], linestyle="--", marker="o", label="Signal MC", ax=myv.ax[0]
    )
    myv.ax[0].set(ylabel="North [(m/s)$^2$]", yscale="log")
    myv.ax[0].minorticks_on()
    myv.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFve"], marker=">", ax=myv.ax[1])
    sns.lineplot(x=corr["CNo"], y=corr["MCve"], marker="o", ax=myv.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["KFve"], linestyle="--", marker=">", ax=myv.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["MCve"], linestyle="--", marker="o", ax=myv.ax[1])
    myv.ax[1].set(ylabel="East [(m/s)$^2$]", yscale="log")
    myv.ax[1].minorticks_on()
    myv.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFvd"], marker=">", ax=myv.ax[2])
    sns.lineplot(x=corr["CNo"], y=corr["MCvd"], marker="o", ax=myv.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["KFvd"], linestyle="--", marker=">", ax=myv.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["MCvd"], linestyle="--", marker="o", ax=myv.ax[2])
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
    sns.lineplot(x=corr["CNo"], y=corr["KFr"], marker=">", label="Correlator KF", ax=mya.ax[0])
    sns.lineplot(x=corr["CNo"], y=corr["MCr"], marker="o", label="Correlator MC", ax=mya.ax[0])
    sns.lineplot(
        x=sig["CNo"], y=sig["KFr"], linestyle="--", marker=">", label="Signal KF", ax=mya.ax[0]
    )
    sns.lineplot(
        x=sig["CNo"], y=sig["MCr"], linestyle="--", marker="o", label="Signal MC", ax=mya.ax[0]
    )
    mya.ax[0].set(ylabel="Roll [deg$^2$]", yscale="log")
    mya.ax[0].minorticks_on()
    mya.ax[0].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFp"], marker=">", ax=mya.ax[1])
    sns.lineplot(x=corr["CNo"], y=corr["MCp"], marker="o", ax=mya.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["KFp"], linestyle="--", marker=">", ax=mya.ax[1])
    sns.lineplot(x=sig["CNo"], y=sig["MCp"], linestyle="--", marker="o", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [deg$^2$]", yscale="log")
    mya.ax[1].minorticks_on()
    mya.ax[1].grid(which="minor", alpha=0.4)
    sns.lineplot(x=corr["CNo"], y=corr["KFy"], marker=">", ax=mya.ax[2])
    sns.lineplot(x=corr["CNo"], y=corr["MCy"], marker="o", ax=mya.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["KFy"], linestyle="--", marker=">", ax=mya.ax[2])
    sns.lineplot(x=sig["CNo"], y=sig["MCy"], linestyle="--", marker="o", ax=mya.ax[2])
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

    # open plots
    win.show()
    sys.exit(app.exec())
