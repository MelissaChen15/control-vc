"""
Utils

Adapted from:
    https://github.com/jik876/hifi-gan
    https://github.com/facebookresearch/speech-resynthesis
"""


import glob
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt

import librosa
import soundfile as sf
from scipy.io.wavfile import write


def denorm_f0(f0, std_, mean_):
    ii = (f0 != 0)
    f0[ii] *= std_
    f0[ii] += mean_

    return f0

def norm_f0(f0, std_=None, mean_=None):
    ii = (f0 != 0)
    if not std_ or not mean_:
        mean_, std_ = f0[ii].mean(), f0[ii].std()
    f0[ii] -= mean_
    f0[ii] /= std_

    return f0

def save_audio(filename, audio, sample_rate):
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(filename, sample_rate, audio)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    """
    Weight normalization to all conv layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

def get_wavlist4hubert(in_dir, out_file, ext=".wav"):
    with open(out_file, "w") as out:
        out.write(str(in_dir) + "\n")
        for utt in Path(in_dir).glob("**/*" + ext):
            wav = sf.SoundFile(utt)
            line = utt.parent.name + "/" + utt.name + "\t" + str(wav.frames) + "\n"
            out.write(line)

def gen_curve(n_segments, mode="fsf"):
    MAX = 1.5
    MIN = 0.2
    CONST = 1.0
    rates = [0.0] * n_segments

    if mode == "constant":
        rates = [CONST] * n_segments
    elif mode == "fsf":# fast-slow-fast(0.5-2-0.5)
        split = int(n_segments / 3)
        for i in range(split): rates[i] = MAX
        for i in range(split,split*2): rates[i] = MIN
        for i in range(split*2, n_segments): rates[i] = MAX
    elif mode == "parabola":
        x = np.array(range(n_segments))
        a = 4 * (MIN - MAX) / (n_segments * n_segments)
        rates = a * (x - n_segments / 2)**2 + MAX
    elif mode == "down":
        x = np.array(range(n_segments))
        rates = (MIN - MAX) / n_segments * x + MAX
    elif mode == "up":
        x = np.array(range(n_segments))
        rates = (MAX - MIN) / n_segments * x + MIN
    elif mode == "question":
        k = 4 * (MAX - 1) / n_segments
        for x in range(int(n_segments*0.75), n_segments): 
            rates[x] = max(1.0, k*x - 3*MAX + 4)
    elif mode == "stress":
        k = 4 * (1 - MAX) / n_segments
        for x in range(int(n_segments*0.5), int(n_segments*0.75)): 
            rates[x] =  k*x + 3*MAX - 2
    else:
        raise NotImplementedError   
    return rates

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
