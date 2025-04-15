import numpy as np
from satutils import ephemeris, atmosphere
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML


def ParseConfig(filename: Path | str) -> dict:
    yaml = YAML()
    with open(filename, "r") as f:
        conf = dict(yaml.load(f))
    return conf


def ParseEphem(
    filename: str, is_numpy: bool = False
) -> tuple[np.ndarray[ephemeris.KeplerElements], np.ndarray[atmosphere.KlobucharElements]]:
    eph_type = np.dtype(
        [
            ("id", np.uint8),
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
    if is_numpy:
        return data

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


def ParseNavSimStates(filename: str) -> pd.DataFrame:
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
    df = pd.DataFrame.from_records(data)
    # df.loc[:, "t"] /= 1000.0 # s to ms
    return df


def ParseCorrelatorSimLogs(
    directory: str, is_array: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray[pd.DataFrame]]:
    nav_log_type = np.dtype(
        [
            ("t", np.double),
            ("tR", np.double),
            ("Lat", np.double),
            ("Lon", np.double),
            ("H", np.double),
            ("vN", np.double),
            ("vE", np.double),
            ("vD", np.double),
            ("qw", np.double),
            ("qx", np.double),
            ("qy", np.double),
            ("qz", np.double),
            ("Bias", np.double),
            ("Drift", np.double),
            ("P0", np.double),
            ("P1", np.double),
            ("P2", np.double),
            ("P3", np.double),
            ("P4", np.double),
            ("P5", np.double),
            ("P6", np.double),
            ("P7", np.double),
            ("P8", np.double),
            ("P9", np.double),
            ("P10", np.double),
        ],
    )

    err_log_type = np.dtype(
        [
            ("t", np.double),
            ("tR", np.double),
            ("N", np.double),
            ("E", np.double),
            ("D", np.double),
            ("vN", np.double),
            ("vE", np.double),
            ("vD", np.double),
            ("Roll", np.double),
            ("Pitch", np.double),
            ("Yaw", np.double),
            ("Bias", np.double),
            ("Drift", np.double),
        ],
    )

    if is_array:
        channel_log_type = np.dtype(
            [
                ("t", np.double),
                ("ToW", np.double),
                ("az", np.double),
                ("el", np.double),
                ("phase", np.double),
                ("omega", np.double),
                ("chip", np.double),
                ("chip_rate", np.double),
                ("cno", np.double),
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
                ("ToW", np.double),
                ("az", np.double),
                ("el", np.double),
                ("phase", np.double),
                ("omega", np.double),
                ("chip", np.double),
                ("chip_rate", np.double),
                ("cno", np.double),
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
                pd.DataFrame.from_records(
                    np.asfortranarray(np.fromfile(path, dtype=channel_log_type))
                )
            )
        elif "Nav" in pathstr:
            nav = pd.DataFrame.from_records(
                np.asfortranarray(np.fromfile(path, dtype=nav_log_type))
            )
        elif "Err" in pathstr:
            err = pd.DataFrame.from_records(
                np.asfortranarray(np.fromfile(path, dtype=err_log_type))
            )
    return nav, err, channels


def ParseSturdrLogs(
    directory: str, is_array: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray[pd.DataFrame]]:
    nav_log_type = np.dtype(
        [
            ("MsElapsed", np.uint64),
            ("Week", np.uint16),
            ("ToW", np.double),
            ("Lat", np.double),
            ("Lon", np.double),
            ("Alt", np.double),
            ("vN", np.double),
            ("vE", np.double),
            ("vD", np.double),
            ("qw", np.double),
            ("qx", np.double),
            ("qy", np.double),
            ("qz", np.double),
            ("cb", np.double),
            ("cd", np.double),
            ("Lat_Var", np.double),
            ("Lon_Var", np.double),
            ("Alt_Var", np.double),
            ("vN_Var", np.double),
            ("vE_Var", np.double),
            ("vD_Var", np.double),
            ("Roll_Var", np.double),
            ("Pitch_Var", np.double),
            ("Yaw_Var", np.double),
            ("cb_Var", np.double),
            ("cd_Var", np.double),
        ],
    )

    if is_array:
        channel_log_type = np.dtype(
            [
                ("ChannelNum", np.uint8),
                ("Constellation", np.uint8),
                ("Signal", np.uint8),
                ("SVID", np.uint8),
                ("ChannelStatus", np.uint8),
                ("TrackingStatus", np.uint8),
                ("Week", np.uint16),
                ("ToW", np.double),
                ("CNo", np.double),
                ("Doppler", np.double),
                ("CodePhase", np.double),
                ("CarrierPhase", np.double),
                ("IE", np.double),
                ("IP", np.double),
                ("IL", np.double),
                ("QE", np.double),
                ("QP", np.double),
                ("QL", np.double),
                ("IP1", np.double),
                ("IP2", np.double),
                ("QP1", np.double),
                ("QP2", np.double),
                ("DllDisc", np.double),
                ("PllDisc", np.double),
                ("FllDisc", np.double),
                ("IP_reg_0", np.double),
                ("IP_reg_1", np.double),
                ("IP_reg_2", np.double),
                ("IP_reg_3", np.double),
                ("QP_reg_0", np.double),
                ("QP_reg_1", np.double),
                ("QP_reg_2", np.double),
                ("QP_reg_3", np.double),
            ]
        )
    else:
        channel_log_type = np.dtype(
            [
                ("ChannelNum", np.uint8),
                ("Constellation", np.uint8),
                ("Signal", np.uint8),
                ("SVID", np.uint8),
                ("ChannelStatus", np.uint8),
                ("TrackingStatus", np.uint8),
                ("Week", np.uint16),
                ("ToW", np.double),
                ("CNo", np.double),
                ("Doppler", np.double),
                ("CodePhase", np.double),
                ("CarrierPhase", np.double),
                ("IE", np.double),
                ("IP", np.double),
                ("IL", np.double),
                ("QE", np.double),
                ("QP", np.double),
                ("QL", np.double),
                ("IP1", np.double),
                ("IP2", np.double),
                ("QP1", np.double),
                ("QP2", np.double),
                ("DllDisc", np.double),
                ("PllDisc", np.double),
                ("FllDisc", np.double),
            ]
        )

    pathlist = sorted([str(path) for path in Path(directory).glob("**/*.bin")])
    channels = []
    for path in pathlist:
        pathstr = str(path)
        if "Ch" in pathstr:
            channels.append(
                pd.DataFrame.from_records(
                    np.asfortranarray(np.fromfile(path, dtype=channel_log_type))
                )
            )
        elif "Nav" in pathstr:
            data = np.asfortranarray(np.fromfile(path, dtype=nav_log_type))
            nav = pd.DataFrame(
                data[
                    [
                        "MsElapsed",
                        "Week",
                        "ToW",
                        "Lat",
                        "Lon",
                        "Alt",
                        "vN",
                        "vE",
                        "vD",
                        "qw",
                        "qx",
                        "qy",
                        "qz",
                        "cb",
                        "cd",
                    ]
                ]
            )
            var = pd.DataFrame(
                data[
                    [
                        "MsElapsed",
                        "Week",
                        "ToW",
                        "Lat_Var",
                        "Lon_Var",
                        "Alt_Var",
                        "vN_Var",
                        "vE_Var",
                        "vD_Var",
                        "Roll_Var",
                        "Pitch_Var",
                        "Yaw_Var",
                        "cb_Var",
                        "cd_Var",
                    ]
                ]
            )

    return nav, var, channels
