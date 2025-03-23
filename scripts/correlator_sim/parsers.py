import numpy as np
from satutils import ephemeris, atmosphere
import pandas as pd
from pathlib import Path
from yaml import safe_load


def ParseConfig(filename: str):
    with open(filename) as f:
        conf = safe_load(f)
    return conf


def ParseEphem(filename: str) -> np.ndarray[ephemeris.KeplerElements]:
    eph_type = np.dtype(
        [
            ("iode", np.double),
            ("iodc", np.double),
            ("toe", np.double),
            ("toc", np.double),
            ("tgd", np.double),
            ("af2", np.double),
            ("af1", np.double),
            ("af0", np.double),
            ("e", np.double),
            ("sqrtA", np.double),
            ("deltan", np.double),
            ("m0", np.double),
            ("omega0", np.double),
            ("omega", np.double),
            ("omegaDot", np.double),
            ("i0", np.double),
            ("iDot", np.double),
            ("cuc", np.double),
            ("cus", np.double),
            ("cic", np.double),
            ("cis", np.double),
            ("crc", np.double),
            ("crs", np.double),
            ("ura", np.double),
            ("health", np.double),
        ],
    )

    data = np.asfortranarray(np.fromfile(filename, dtype=eph_type))
    out = np.empty(data.size, dtype=ephemeris.KeplerElements, order="F")
    out2 = np.empty(data.size, dtype=atmosphere.KlobucharElements, order="F")
    for i in range(data.size):
        out[i] = ephemeris.KeplerElements()
        out[i].iode = data[i]["iode"]
        out[i].iodc = data[i]["iodc"]
        out[i].toe = data[i]["toe"]
        out[i].toc = data[i]["toc"]
        out[i].tgd = data[i]["tgd"]
        out[i].af2 = data[i]["af2"]
        out[i].af1 = data[i]["af1"]
        out[i].af0 = data[i]["af0"]
        out[i].e = data[i]["e"]
        out[i].sqrtA = data[i]["sqrtA"]
        out[i].deltan = data[i]["deltan"]
        out[i].m0 = data[i]["m0"]
        out[i].omega0 = data[i]["omega0"]
        out[i].omega = data[i]["omega"]
        out[i].omegaDot = data[i]["omegaDot"]
        out[i].i0 = data[i]["i0"]
        out[i].iDot = data[i]["iDot"]
        out[i].cuc = data[i]["cuc"]
        out[i].cus = data[i]["cus"]
        out[i].cic = data[i]["cic"]
        out[i].cis = data[i]["cis"]
        out[i].crc = data[i]["crc"]
        out[i].crs = data[i]["crs"]
        out[i].ura = data[i]["ura"]
        out[i].health = data[i]["health"]

        out2[i] = atmosphere.KlobucharElements()
        out2[i].a0 = 0.0
        out2[i].a1 = 0.0
        out2[i].a2 = 0.0
        out2[i].a3 = 0.0
        out2[i].b0 = 0.0
        out2[i].b1 = 0.0
        out2[i].b2 = 0.0
        out2[i].b3 = 0.0

    return out, out2


def ParseNavStates(filename: str) -> pd.DataFrame:
    nav_type = np.dtype(
        [
            ("t", np.double),
            ("lat", np.double),
            ("lon", np.double),
            ("h", np.double),
            ("vn", np.double),
            ("ve", np.double),
            ("vd", np.double),
            ("r", np.double),
            ("p", np.double),
            ("y", np.double),
        ]
    )
    data = np.asfortranarray(np.fromfile(filename, dtype=nav_type))
    return pd.DataFrame(data)


def ParseCorrelatorSimLogs(
    directory: str,
    is_array: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray[pd.DataFrame]]:
    nav_log_type = np.dtype(
        [
            ("t", np.double),
            ("tow", np.double),
            ("lat", np.double),
            ("lon", np.double),
            ("h", np.double),
            ("vn", np.double),
            ("ve", np.double),
            ("vd", np.double),
            ("r", np.double),
            ("p", np.double),
            ("y", np.double),
            ("cb", np.double),
            ("cd", np.double),
        ],
    )

    err_log_type = np.dtype(
        [
            ("t", np.double),
            ("tow", np.double),
            ("lat", np.double),
            ("lon", np.double),
            ("h", np.double),
            ("vn", np.double),
            ("ve", np.double),
            ("vd", np.double),
            ("r", np.double),
            ("p", np.double),
            ("y", np.double),
            ("cb", np.double),
            ("cd", np.double),
        ],
    )

    if is_array:
        channel_log_type = np.dtype(
            [
                ("t", np.double),
                ("tow", np.double),
                ("az", np.double),
                ("el", np.double),
                ("true_phase", np.double),
                ("true_omega", np.double),
                ("true_chip", np.double),
                ("true_chip_rate", np.double),
                ("true_cno", np.double),
                ("est_phase", np.double),
                ("est_omega", np.double),
                ("est_chip", np.double),
                ("est_chip_rate", np.double),
                ("est_cno", np.double),
                ("IE", np.double),
                ("QE", np.double),
                ("IP", np.double),
                ("QP", np.double),
                ("IL", np.double),
                ("QL", np.double),
                ("IP1", np.double),
                ("QP1", np.double),
                ("IP2", np.double),
                ("QP2", np.double),
                ("IP_reg_0", np.double),
                ("QP_reg_0", np.double),
                ("IP_reg_1", np.double),
                ("QP_reg_1", np.double),
                ("IP_reg_2", np.double),
                ("QP_reg_2", np.double),
                ("IP_reg_3", np.double),
                ("QP_reg_3", np.double),
            ],
        )
    else:
        channel_log_type = np.dtype(
            [
                ("t", np.double),
                ("tow", np.double),
                ("az", np.double),
                ("el", np.double),
                ("true_phase", np.double),
                ("true_omega", np.double),
                ("true_chip", np.double),
                ("true_chip_rate", np.double),
                ("true_cno", np.double),
                ("est_phase", np.double),
                ("est_omega", np.double),
                ("est_chip", np.double),
                ("est_chip_rate", np.double),
                ("est_cno", np.double),
                ("IE", np.double),
                ("QE", np.double),
                ("IP", np.double),
                ("QP", np.double),
                ("IL", np.double),
                ("QL", np.double),
                ("IP1", np.double),
                ("QP1", np.double),
                ("IP2", np.double),
                ("QP2", np.double),
            ],
        )

    # pathlist = Path(directory).glob("**/*.bin")
    pathlist = sorted([str(path) for path in Path(directory).glob("**/*.bin")])
    channels = []
    for path in pathlist:
        pathstr = str(path)
        if "Channel" in pathstr:
            channels.append(
                pd.DataFrame(np.asfortranarray(np.fromfile(path, dtype=channel_log_type)))
            )
        elif "Nav" in pathstr:
            nav = pd.DataFrame(np.asfortranarray(np.fromfile(path, dtype=nav_log_type)))
        elif "Err" in pathstr:
            err = pd.DataFrame(np.asfortranarray(np.fromfile(path, dtype=err_log_type)))
        elif "Var" in pathstr:
            var = pd.DataFrame(np.asfortranarray(np.fromfile(path, dtype=err_log_type)))
    return nav, err, var, channels
