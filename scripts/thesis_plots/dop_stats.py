import pandas as pd
import numpy as np
from navtools._navtools_core.frames import ecef2nedDcm
import sys

sys.path.append("scripts")
from correlator_sim.models import ObservableModel
from utils.parsers import ParseNavSimStates, ParseEphem


def average_dop(truth: pd.DataFrame, obs: ObservableModel):
    H = np.zeros((obs.size, 4), order="F")
    H[:, 3] = 1.0
    gdop = np.zeros(int(len(truth) / 100) + 1, order="F")
    pdop = np.zeros(int(len(truth) / 100) + 1, order="F")
    hdop = np.zeros(int(len(truth) / 100) + 1, order="F")
    vdop = np.zeros(int(len(truth) / 100) + 1, order="F")
    tdop = np.zeros(int(len(truth) / 100) + 1, order="F")
    jj = 0
    for kk in range(0, len(truth), 100):
        pos = np.array([truth.loc[kk, "lat"], truth.loc[kk, "lon"], truth.loc[kk, "h"]], order="F")
        vel = np.array([truth.loc[kk, "vn"], truth.loc[kk, "ve"], truth.loc[kk, "vd"]], order="F")
        C_e_n = ecef2nedDcm(pos)
        for ii in range(obs.size):
            obs[ii].UpdateSatState(truth.loc[kk, "tT"])
            obs[ii].CalcRangeAndRate(pos, vel, 0, 0, False)
            H[ii, 0:3] = C_e_n @ obs[ii].EcefUnitVec.copy()
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
    truth = ParseNavSimStates("data/drone_sim.bin")
    elem, klob = ParseEphem("data/gps_skydel_2024_08_23.bin")
    truth["tT"] = truth["t"] + 494998.07 - 0.07
    obs = np.zeros(elem.size, order="F", dtype=ObservableModel)
    for ii in range(elem.size):
        obs[ii] = ObservableModel(elem[ii], klob[ii])
    gdop, pdop, hdop, vdop, tdop = average_dop(truth, obs)
    print("Drone Sim ...")
    print(f"Average GDOP = {gdop}")
    print(f"Average PDOP = {pdop}")
    print(f"Average HDOP = {hdop}")
    print(f"Average VDOP = {vdop}")
    print(f"Average TDOP = {tdop}")
    print()

    # 2. Vehicle Simulation
    truth = ParseNavSimStates("data/ground_sim.bin")
    elem, klob = ParseEphem("data/gps_skydel_2025_02_07.bin")
    truth["tT"] = truth["t"] + 507178.98 - 0.07
    obs = np.zeros(elem.size, order="F", dtype=ObservableModel)
    for ii in range(elem.size):
        obs[ii] = ObservableModel(elem[ii], klob[ii])
    gdop, pdop, hdop, vdop, tdop = average_dop(truth, obs)
    print("Vehicle Sim ...")
    print(f"Average GDOP = {gdop}")
    print(f"Average PDOP = {pdop}")
    print(f"Average HDOP = {hdop}")
    print(f"Average VDOP = {vdop}")
    print(f"Average TDOP = {tdop}")
    print()

    # 3. Live-sky data
    obs = np.delete(obs, [10, 5])
    gdop, pdop, hdop, vdop, tdop = average_dop(truth, obs)
    print("Like-sky ...")
    print(f"Average GDOP = {gdop}")
    print(f"Average PDOP = {pdop}")
    print(f"Average HDOP = {hdop}")
    print(f"Average VDOP = {vdop}")
    print(f"Average TDOP = {tdop}")
    print()
