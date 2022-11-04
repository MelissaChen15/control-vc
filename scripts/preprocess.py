"""
Audio preprocess for training and inference

Reference:
    https://github.com/facebookresearch/speech-resynthesis
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import librosa
from tqdm import tqdm

from psola import from_file_to_file, vocode

import os
import sys
sys.path.append(os.path.realpath('.'))
from utils import gen_curve


def change_rhythm_from_curve(audio, sr, mode="up", seg_size=0.16, silent_front=0.48, silent_end=0.32):
    seg_size = int(seg_size * sr)
    silent_front = int(silent_front / seg_size)
    silent_end = int(silent_end / seg_size)
    N = len(audio)

    if N % seg_size != 0:
        padding = int((N // seg_size + 1) * seg_size - N)
        audio = np.append(audio, [0.0]*padding)
        N = len(audio)
    assert(N % seg_size == 0)
    n_segments = int(N // seg_size - silent_front - silent_end)
    
    rates = [1.0] * silent_front + list(gen_curve(n_segments, mode)) + [1.0] * silent_end

    output_audio = []
    for i in range(n_segments):
        segment = audio[i*seg_size: (i+1)*seg_size]
        output_audio.append(vocode(audio=segment, sample_rate=sr, constant_stretch=rates[i]))

    output_audio = np.hstack(output_audio)
    
    return output_audio


def pad_data(p, out_dir, trim=False, pad=True, rhythm_cruve=True, rhythm_mode="up", keep_folder=False, postfix=".wav"):
    """
    resample to 16k
    (optional) trim silence under 20 db
    (optional) pad to frames in 12.5 ms
    """
    data, sr = sf.read(p)
    if sr != 16000:
        data = resampy.resample(data, sr, 16000)
        sr = 16000

    if rhythm_cruve:
        data = change_rhythm_from_curve(data, sr, mode=rhythm_mode)
    
    if trim:
        data, _ = librosa.effects.trim(data, 20) # trim silence(below 20dB)

    if pad:
        if data.shape[0] % 1280 != 0:
            data = np.pad(data, (0, 1280 - data.shape[0] % 1280), mode='constant',
                          constant_values=0)
        assert data.shape[0] % 1280 == 0

    outname = "_".join([p.stem, rhythm_mode]) + postfix if rhythm_cruve else p.stem + postfix
    if keep_folder:
        outpath = out_dir / p.parent.name / outname
    else:
        outpath = out_dir / outname
    outpath.parent.mkdir(exist_ok=True, parents=True)
    sf.write(outpath, data, sr)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('--pad', action='store_true')
    parser.add_argument('--rhythm_cruve', action='store_true')
    parser.add_argument('--rhythm_mode', type=str, default='up')
    parser.add_argument('--postfix', type=str, default='.wav')
    parser.add_argument('--keepfolder', action='store_true') # keep subdirectories for each speaker
    args = parser.parse_args()

    files = list(Path(args.srcdir).glob(f'**/*{args.postfix}'))
    out_dir = Path(args.outdir)

    pad_data_ = partial(pad_data, out_dir=out_dir, rhythm_cruve=args.rhythm_cruve, rhythm_mode=args.rhythm_mode, 
                        trim=args.trim, pad=args.pad, keep_folder=args.keepfolder, postfix=args.postfix)
    with Pool(40) as p:
        rets = list(tqdm(p.imap(pad_data_, files), total=len(files)))


if __name__ == '__main__':
    main()
