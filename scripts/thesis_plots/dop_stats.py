import pandas as pd
import numpy as np
from navtools._navtools_core.frames import ecef2nedDcm
from pickle import load
import sys

sys.path.append("scripts/correlator_sim")
from vt import TruthObservables


def average_dop(truth: TruthObservables, bad_idx: list = []):
    H = np.zeros((len(truth.az) - len(bad_idx), 4), order="F")
    H[:, 3] = 1.0
    tR = np.arange(truth.tR.c[0], truth.tR.c[-1], 1.0)
    gdop = np.zeros(tR.size, order="F")
    pdop = np.zeros(tR.size, order="F")
    hdop = np.zeros(tR.size, order="F")
    vdop = np.zeros(tR.size, order="F")
    tdop = np.zeros(tR.size, order="F")
    jj = 0
    idx = np.arange(0, len(truth.az))
    idx = np.delete(idx, bad_idx)
    for kk in range(0, tR.size):
        ll = 0
        for ii in idx:
            az = truth.az[ii][0](tR[kk])
            el = truth.el[ii][0](tR[kk])
            H[ll, 0] = np.cos(az) * np.cos(el)
            H[ll, 1] = np.sin(az) * np.cos(el)
            H[ll, 2] = -np.sin(el)
            ll += 1
        D = np.diag(np.linalg.inv(H.T @ H))
        tdop[jj] = np.sqrt(D[3])
        vdop[jj] = np.sqrt(D[2])
        hdop[jj] = np.sqrt(D[0] + D[1])
        pdop[jj] = np.sqrt(D[0] + D[1] + D[2])
        gdop[jj] = np.sqrt(D[0] + D[1] + D[2] + D[3])
        jj += 1
    return gdop.mean(), pdop.mean(), hdop.mean(), vdop.mean(), tdop.mean()


if __name__ == "__main__":
    # 1. Drone Simulation
    with open(
        "/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/drone-sim/splines/Run1.bin", "rb"
    ) as file:
        truth = load(file)
    gdop, pdop, hdop, vdop, tdop = average_dop(truth)
    print("Drone Sim ...")
    print(f"Average GDOP = {gdop:.3f}")
    print(f"Average PDOP = {pdop:.3f}")
    print(f"Average HDOP = {hdop:.3f}")
    print(f"Average VDOP = {vdop:.3f}")
    print(f"Average TDOP = {tdop:.3f}")
    print()

    # 2. Vehicle Simulation
    with open(
        "/media/daniel/Sturdivant/Thesis-Data/Correlator-Sim/ground-sim/splines/Run1.bin", "rb"
    ) as file:
        truth = load(file)
    gdop, pdop, hdop, vdop, tdop = average_dop(truth)
    print("Vehicle Sim ...")
    print(f"Average GDOP = {gdop:.3f}")
    print(f"Average PDOP = {pdop:.3f}")
    print(f"Average HDOP = {hdop:.3f}")
    print(f"Average VDOP = {vdop:.3f}")
    print(f"Average TDOP = {tdop:.3f}")
    print()

    # 3. Live-sky data
    gdop, pdop, hdop, vdop, tdop = average_dop(truth, [5, 10])
    print("Like-sky ...")
    print(f"Average GDOP = {gdop:.3f}")
    print(f"Average PDOP = {pdop:.3f}")
    print(f"Average HDOP = {hdop:.3f}")
    print(f"Average VDOP = {vdop:.3f}")
    print(f"Average TDOP = {tdop:.3f}")
    print()
