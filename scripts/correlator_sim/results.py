from parsers import ParseCorrelatorSimLogs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nav, err, var, channels = ParseCorrelatorSimLogs("results/VT_ARRAY_CORRELATOR_SIM/1/")

    # plot channel data
    COLORS = ["#a2e3b8", "#a52a2a", "#100c08"]
    sns.set_theme(
        font="Times New Roman",
        # context="talk",
        context="paper",
        palette=sns.color_palette(COLORS),
        style="ticks",
        rc={
            "axes.grid": True,
            "lines.linewidth": 2,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        },
        font_scale=1.5,
    )
    f0, ax0 = plt.subplots()
    sns.lineplot(x=channels[0]["t"], y=channels[0]["est_cno"], label="Estimated", ax=ax0)
    # sns.lineplot(x=channels[0]["t"], y=channels[0]["est_cno_bs"], label="Beam Steered", ax=ax0)
    sns.lineplot(x=channels[0]["t"], y=channels[0]["true_cno"], label="Truth", ax=ax0)
    ax0.set(xlabel=r"t [s]", ylabel=r"C/N\textsubscript{0} [dB-Hz]")
    f0.tight_layout()

    # plot error statistics
    COLORS = ["#100c08", "#a52a2a", "#a52a2a"]
    sns.set_theme(
        font="Times New Roman",
        # context="talk",
        context="paper",
        palette=sns.color_palette(COLORS),
        style="ticks",
        rc={
            "axes.grid": True,
            "lines.linewidth": 2,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        },
        font_scale=1.5,
    )
    f1, ax1 = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    sns.lineplot(x=err["t"], y=err["lat"], label=r"$\mu$", ax=ax1[0])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["lat"]), label=r"$3\sigma$", ax=ax1[0])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["lat"]), ax=ax1[0])
    ax1[0].set(ylim=[-3, 3], ylabel="North [m]")
    sns.lineplot(x=err["t"], y=err["lon"], ax=ax1[1])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["lon"]), ax=ax1[1])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["lon"]), ax=ax1[1])
    ax1[1].set(ylim=[-3, 3], ylabel="East [m]")
    sns.lineplot(x=err["t"], y=err["h"], ax=ax1[2])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["h"]), ax=ax1[2])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["h"]), ax=ax1[2])
    ax1[2].set(ylim=[-3, 3], ylabel="Down [m]", xlabel="t [s]")
    f1.tight_layout()

    f2, ax2 = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    sns.lineplot(x=err["t"], y=err["r"], label=r"$\mu$", ax=ax2[0])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["r"]), label=r"$3\sigma$", ax=ax2[0])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["r"]), ax=ax2[0])
    ax2[0].set(ylim=[-3, 3], ylabel="Roll [deg]")
    sns.lineplot(x=err["t"], y=err["p"], ax=ax2[1])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["p"]), ax=ax2[1])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["p"]), ax=ax2[1])
    ax2[1].set(ylim=[-3, 3], ylabel="Pitch [m]")
    sns.lineplot(x=err["t"], y=err["y"], ax=ax2[2])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["y"]), ax=ax2[2])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["y"]), ax=ax2[2])
    ax2[2].set(ylabel="Yaw [m]", xlabel="t [s]")
    f2.tight_layout()

    plt.show()

    print()
