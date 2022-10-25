"""
Datasets

Reference:
    https://github.com/jik876/hifi-gan
    https://github.com/facebookresearch/speech-resynthesis
"""

import random
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0 # for 16 bit wavs


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate) # create a signal project
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec) # log((x>thre) * C) -> log compression

    return spec


def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def parse_manifest(manifest):
    '''
    Parsing the preprocessed content embeddings
    '''
    audio_files = []
    codes = []
    embeds = []
    rates = []
    boundaries = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                else:
                    k = 'hubert'
                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
                
                if 'spk_embed' in sample:
                    embeds += [torch.FloatTensor([float(x) for x in sample['spk_embed'].split(' ')]).numpy()]
                if 'rate' in sample:
                    rates += [torch.LongTensor([int(x) for x in sample['rate'].split(' ')]).numpy()]
                if 'boundary' in sample:
                    boundaries += [torch.FloatTensor([float(x) for x in sample['boundary'].split(' ')]).numpy()]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes, embeds, rates, boundaries


def get_dataset_filelist(h):
    train_files, train_codes,train_embeds,train_rates, train_boundaries = parse_manifest(h.input_training_file)
    val_files, val_codes,val_embeds, val_rates, val_boundaries = parse_manifest(h.input_validation_file)

    return (train_files, train_codes,train_embeds,train_rates, train_boundaries), (val_files, val_codes,val_embeds, val_rates, val_boundaries)


def parse_speaker(path, method):
    '''
    get speaker id from the input wav path
    '''
    if type(path) == str:
        path = Path(path)

    if method == 'parent_name':
        return path.parent.name
    elif method == 'parent_parent_name':
        return path.parent.parent.name
    elif method == '_':
        return path.name.split('_')[0]
    elif method == 'single':
        return 'A'
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


class CodeDataset(torch.utils.data.Dataset):
    '''
    Returns the dataset with code(content embedding)

    Inputs:

        training set:
        - training_files: wav path and content embeddings
        - n_cache_reuse: cache used for get_item
        - device: cuda device
        - fmax_loss: max frequency for loss calculation
        - split: true if there is a train/val split
        - vqvae: if the model is vqvae


        segment and feats extract(non-Mel):
        - segment_size: size of frames. Set to the max size of the utterences if segment_size is not specified by config files.
        - code_hop_size: hop size for processing that is not for extracting Mels
        - pad: pad sequence for complete segments, in samples
        - n_fft, num_mels,hop_size, win_size, sampling_rate, fmin, fmax: params for Mels calcualtion

        F0 handling:
        - f0: use yaapt to get f0 sequence if true
        - f0_interp: when calling yaapt, replace unvoiced segments with the interpolation from the adjacent voiced segments edges
        - f0_stats: path where f0_stats.th is stored
        - f0_normalize: if true, f0 = (f0-mean)/std. use averaged mean and std if the spk is not found in f0_stats
        - f0_median: if ture, use normalized median number to fill locs where f0=0
        - f0_feats:  if true, add f0 stats(mean and std) to feature sets
            
    Returns:
        (feats, gt_audio, gt_wavpath, gt_mel-spectrogam)

        feats["code", "f0", "spk_name", "spk_embed","f0_stats"]
        code for content embedding
        f0 is extracted and processed f0 sequence
        spk is reasigned speaker id
        f0_stats is (mean, std) 
    '''
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, f0=None, pad=None,
                 f0_stats=None, f0_normalize=False, f0_feats=False, f0_median=False,
                 f0_interp=False, vqvae=False, random_sample=False, rate=False, boundary=False):
        # self.audio_files, self.codes = training_files
        self.audio_files, self.codes, self.spk_embeds, self.rates, self.bds = training_files # codes are content embeddings
        # print(self.audio_files[0], self.codes[0], self.spk_embeds[0]) #correct
        # exit(1)
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.random_sample = random_sample
        self.rate = rate
        self.boundary = boundary
        self.f0 = f0
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = None
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        if f0_stats:
            self.f0_stats = np.load(f0_stats, allow_pickle=True)
        self.pad = pad
        spkrs = [parse_speaker(f, "_") for f in self.audio_files]
        spkrs = list(set(spkrs))
        spkrs.sort()
        self.spkrs = spkrs

    def _sample_interval(self, seqs, seq_len=None):
        '''
        returns original sequence if self.segment_size = N
        else, returns segment with length = self.segment_size, start from a random number in range(interval_start, interval_end)
        '''

        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs
    
    def _sample_code(self, code, criterion=None):
        if criterion == "unique": # cancel all repetitive codes
            return self._get_unique_code(code)
        if criterion == "random":
            unique_code = self._get_unique_code(code)
            duration, status = self._get_random_duration(len(code), len(unique_code))
            if status == -1:
                return [None]
            return self._code_expansion(unique_code, duration, len(code))

    def _code_expansion(self, code, duration, original_length):
        assert len(code) == len(duration), "length not match"

        new_code = []
        for _, (c, d) in enumerate(zip(code, duration)):
            new_code.append(np.repeat(c, d))
        new_code = np.hstack(new_code)

        assert(len(new_code) == original_length)

        return new_code

    def _get_unique_code(self, code):
        new_code = []
 
        for i in range(len(code)):
            if i == 0: 
                new_code.append(code[i])
            else:
                if np.abs(code[i] - code[i-1]) > 0:
                    new_code.append(code[i])
        return np.array(new_code)
    
    def _get_random_duration(self, original_length, unique_length):
        # generate random durations
        mu, sigma = 2, 1 # get those numbers from phoneme duration stats
        sample = np.random.normal(mu, sigma, unique_length)
        duration = []
        for d in sample:
            if d < 1: d += 2 * (mu-d)
            duration.append(int(d))

        # adjust to match original length
        new_length = np.sum(duration)
        try:
            if new_length != original_length:
                diff = original_length - new_length
                sign = 1 if diff > 0 else -1
                diff = np.abs(diff)
                loc = -1
                while(diff != 0):
                    if sign == -1:
                        buff = duration[loc] - 1
                    else:
                        buff = random.randint(1, 20)
                    adjust = buff if buff < diff else diff
                    if loc == 0:
                        adjust = diff
                    duration[loc] += sign*adjust
                    diff -= adjust
                    loc -= 1
        except Exception as e:
            print("code resample failed. ", "original error msg:", e)
            return [], -1
        assert duration[-1] >= 1
        assert np.sum(duration) == original_length

        return duration, 1


    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95 # input audio is normalized to [-1,1] and * 0.95 to avoid clipping
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1
        

        # match audio and code length
        if self.vqvae:
            code_length = audio.shape[0] // self.code_hop_size
        elif self.rate:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0], self.rates[index].shape[0])
            code = self.codes[index][:code_length]
            rate = self.rates[index][:code_length]
        else:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
            code = self.codes[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]
        assert self.vqvae or audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"
        if self.rate:
            assert code.shape[0] == rate.shape[0], "Code rate mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            if not self.vqvae:
                code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"


        if self.boundary:
            raise NotImplementedError
            feats['boundary'] = self.boundaries[index]
        
        # random sample audio and code 
        if self.vqvae:
            audio = self._sample_interval([audio])[0]
        elif self.rate:
            audio, code, rate = self._sample_interval([audio, code, rate])
        else:
            audio, code = self._sample_interval([audio, code])
        

        if self.random_sample:
            code_buffer = self._sample_code(code, criterion="random")
            if code_buffer[0] is not None: code = code_buffer


        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        if self.vqvae:
            feats = {
                "code": audio.view(1, -1).numpy()
            }
        else:
            feats = {"code": code.squeeze()}
    
        if self.rate:
            feats['rate'] = rate

        if self.f0:
            try:
                f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
            except:
                print(filename, "failed to get f0 from yaapt, using zero f0s.")
                f0 = np.zeros((1, 1, audio.shape[-1] // 80))
            f0 = f0.astype(np.float32)
            feats['f0'] = f0.squeeze(0)

        feats['spkr_name'], feats['spk_embed'] = self._get_spkr(index)
        
        if self.f0_normalize:
            spkr_name, _ = self._get_spkr(index)
            if self.f0_stats is not None:
                if spkr_name not in self.f0_stats:
                    mean = self.f0_stats['f0_mean']
                    std = self.f0_stats['f0_std']
                else:
                    mean = self.f0_stats[spkr_name]['f0_mean']
                    std = self.f0_stats[spkr_name]['f0_std']
            else: # if f0_stats is not given
                mean =  np.mean(feats['f0'])
                std =  np.std(feats['f0'])
            ii = feats['f0'] != 0 # the start location where f0 is not zero

            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
                feats['f0'][~ii] = (feats['f0'][~ii] - mean) / std # use normalized median number to fill locs where f0=0

            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()
        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], "_")
        spk_embed = torch.FloatTensor(self.spk_embeds[idx]).numpy()
        return spkr_name, spk_embed

    def __len__(self):
        return len(self.audio_files)


class F0Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate,
                 split=True, n_cache_reuse=1, device=None,
                 pad=None, f0_stats=None, f0_normalize=False, f0_feats=False,
                 f0_median=False, f0_interp=False, vqvae=False):
        self.audio_files, self.codes, self.spk_embeds, self.rates, self.bds = training_files # codes are content embeddings
        random.seed(1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = None
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        if f0_stats:
            self.f0_stats = np.load(f0_stats, allow_pickle=True)
        self.pad = pad
        
    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        audio = self._sample_interval([audio])[0]

        feats = {}
        try:
            f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
        except:
            print(filename, "failed to get f0 from yaapt, using zero f0s.")
            f0 = np.zeros((1, 1, audio.shape[-1] // 80))
        f0 = f0.astype(np.float32)
        feats['f0'] = f0.squeeze(0)

        feats['spkr_name'], feats['spk_embed'] = self._get_spkr(index)


        if self.f0_normalize:
            spkr_name, _ = self._get_spkr(index)
            if self.f0_stats is not None:
                if spkr_name not in self.f0_stats:
                    mean = self.f0_stats['f0_mean']
                    std = self.f0_stats['f0_std']
                else:
                    mean = self.f0_stats[spkr_name]['f0_mean']
                    std = self.f0_stats[spkr_name]['f0_std']
            else: # if f0_stats is not given
                mean =  np.mean(feats['f0'])
                std =  np.std(feats['f0'])
            ii = feats['f0'] != 0 # the start location where f0 is not zero

            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
                feats['f0'][~ii] = (feats['f0'][~ii] - mean) / std

            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()

        return feats, feats['f0'], str(filename)

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], "_")
        spk_embed = torch.FloatTensor(self.spk_embeds[idx]).numpy()
        return spkr_name, spk_embed

    def __len__(self):
        return len(self.audio_files)
