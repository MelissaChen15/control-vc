"""
Extract Mel-spectrogram for speaker embeding.

Return shape of: (n_frames, n_mels)
Reference:
    https://github.com/auspicious3000/autovc/blob/master/make_spect.py
"""

from pathlib import Path
import random
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import argparse


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
def extract_mel(in_dir="", out_dir="", ext=".wav"):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    filenames = in_dir.glob(f"**/*{ext}")


    for filename in sorted(filenames):
        prng = RandomState(random.randint(0, 1000)) 
            
        # Read audio file
        x, fs = sf.read(filename)
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)     #(n_frames, n_mels)
        # save spect  
        outname = out_dir / filename.parent.name / f'{filename.stem}.npy'
        outname.parent.mkdir(parents=True, exist_ok=True)
        np.save(outname,S.astype(np.float32), allow_pickle=False)    

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--wavdir', type=Path, default=None)
    parser.add_argument('--meldir', type=Path, default=None)
    parser.add_argument('--ext', type=str, default=".wav")
    args = parser.parse_args()
    
    extract_mel(args.wavdir, args.meldir, args.ext)
    
if __name__ == "__main__":
    main()
