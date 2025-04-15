import numpy as np
from scipy.signal import decimate
from pathlib import Path
from ruamel.yaml import YAML
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
from time import time

# from subprocess import run
from sturdr import SturDR


# TODO: REDO Run7


def mp_combine_int16_iq_files(args):
    combine_int16_iq_files(*args)


def combine_int16_iq_files(
    sig_file: Path,
    awgn_file: Path,
    out_file: Path,
    factor: int = 2,
    chunksize: int = 2**29,  # 0.5 GB,
):
    """
    Sums two large binary files together while avoiding int16 overflow. It also down-samples the
    signal.
    """
    with open(sig_file, "rb") as fs, open(awgn_file, "rb") as fn, open(out_file, "wb") as fo:
        while True:
            # read chunks
            s_chunk = np.frombuffer(fs.read(chunksize), dtype=np.int16)
            n_chunk = np.frombuffer(fn.read(chunksize), dtype=np.int16)
            if s_chunk.size == 0 and n_chunk.size == 0:
                return

            # pad
            if s_chunk.size < n_chunk.size:
                s_chunk = np.pad(s_chunk, (0, n_chunk.size - s_chunk.size), mode="constant")
            elif n_chunk.size < s_chunk.size:
                n_chunk = np.pad(n_chunk, (0, s_chunk.size - n_chunk.size), mode="constant")

            # sum together (signal is at 80 db and noise is at 74 dB so dividing the signal by 2
            # should properly handle overflow)
            sum_chunk = s_chunk / 2 + n_chunk

            # downsample with iir filter
            i_chunk = np.round(decimate(sum_chunk[0::2], factor, zero_phase=True)).astype(np.int16)
            q_chunk = np.round(decimate(sum_chunk[1::2], factor, zero_phase=True)).astype(np.int16)
            down_chunk = np.vstack((i_chunk, q_chunk)).reshape((-1,), order="F")

            # write output
            fo.write(down_chunk.tobytes())
    return


if __name__ == "__main__":
    t0 = time()
    freeze_support()
    indir = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim")
    outdir = Path("/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim-2")
    yaml_file = Path("config/vt_signal_sim.yaml")

    # load config
    yaml = YAML()
    with open(yaml_file, "r") as yf:
        conf = dict(yaml.load(yf))

        from pprint import pprint

        pprint(conf)

    # loop through each signal power
    for ii, cno in enumerate(range(20, 42, 2)):
        # change yaml output config
        new_outdir = outdir / f"CNo_{cno}_dB"
        conf["out_folder"] = str(new_outdir)

        # setup status bar
        desc = f"\u001b[31;1m[sturdiws]\u001b[0m C/No = {cno} dB ... "

        # loop through each interference seed
        for jj in tqdm(
            range(30),
            desc=desc,
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ncols=120,
        ):
            new_scenario = f"Run{jj:d}"
            (new_outdir / new_scenario).mkdir(parents=True, exist_ok=True)
            conf["scenario"] = new_scenario

            # write changes to yaml file for sturdr to see
            with open(yaml_file, "w") as yf:
                yaml.dump(conf, yf)

            # combine signal and interference files
            with Pool(processes=4) as p:
                args = [
                    (
                        indir / f"CNo_{cno}_dB" / f"Ant-{kk}.bin",
                        indir / "Noise" / f"awgn-{jj}.bin",
                        Path("data") / f"SigSim-{kk}.bin",
                    )
                    for kk in range(4)
                ]
                for _ in p.imap(mp_combine_int16_iq_files, args):
                    # print("done")
                    pass

            # run sturdr
            sdr = SturDR(str(yaml_file))
            sdr.Start()
            # sdr = run(
            #     ["./build/bin/run_sturdr config/vt_signal_sim.yaml"],
            #     shell=True,
            #     text=True,
            # )

    print(f"\u001b[31;1m[sturdiws]\u001b[0m Finished processing in {(time() - t0):.3f} seconds")
