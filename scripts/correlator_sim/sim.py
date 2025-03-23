import numpy as np
from parsers import ParseConfig, ParseEphem, ParseNavStates
from vt import vt, vt_array


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=100)

    conf = ParseConfig("config/vt_correlator_sim.yaml")
    eph, atm = ParseEphem("data/sim_ephem.bin")
    truth = ParseNavStates("data/sim_truth.bin")
    # truth = ParseNavStates("data/sim_truth.bin")[:-57050]
    # vt(conf, eph, atm, truth)
    vt_array(conf, eph, atm, truth, np.random.randint(0, 9_223_372_036_854_775_807))
