"""
Calculte f0 stats for inference
"""
import os
import numpy as np
import torch
import pickle
import argparse
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

import soundfile as sf
from tqdm import tqdm


def get_yaapt_f0(audio, rate=16000):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    y_pad = np.pad(audio.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
    signal = basic.SignalObj(y_pad, rate)
    pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                            'tda_frame_length': 25.0})

    return pitch.samp_values[None, None, :]

def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate


def infer(srcdir, outdir):
    sub_dirs = list(srcdir.glob("*"))

    f0_stats = {}
    means = []
    stds = []
    print('Calculating f0 stats...')
    for sub_dir in tqdm(sorted(sub_dirs), total=len(sub_dirs)):
        speaker =  sub_dir.name
        filenames = list(sub_dir.glob("*"))
        num_utts = len(filenames)
        
        f0s = []
        for filename in filenames:
            wav, _ = load_audio(filename)
            f0 = get_yaapt_f0(wav)
            f0s.append(f0.squeeze())     
        f0s = np.hstack(f0s)

        ii = f0s != 0
        mean = np.mean(f0s[ii])
        std = np.std(f0s[ii])
        means.append(mean)
        stds.append(std)

        f0_stats[speaker] = {
            "f0_mean" : mean,
            "f0_std" : std,
            }

    f0_stats["mean"] = np.mean(means)
    f0_stats["std"] = np.mean(stds)
    
    outname = os.path.join(outdir, 'f0_stats.pkl')
    with open(outname, "wb") as f:
        pickle.dump(f0_stats, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()

    infer(args.srcdir, args.outdir)

if __name__ == "__main__":
    main()