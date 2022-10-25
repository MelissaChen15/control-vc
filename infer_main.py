"""
Main inference loop

Reference:
    https://github.com/jik876/hifi-gan
    https://github.com/facebookresearch/speech-resynthesis
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
import torch
torch.set_printoptions(profile="full")

from dataset import CodeDataset, parse_manifest,  \
                    mel_spectrogram, MAX_WAV_VALUE
from utils import AttrDict, load_checkpoint, scan_checkpoint, \
                    save_audio, denorm_f0, norm_f0,  gen_curve
from models import CodeGenerator

h = None
device = None


def stream(message):
    sys.stdout.write(f"\r{message}")

def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf

def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global f0_stats
    global spkrs_emb
    global dataset
    global spkr_dataset
    global idx
    global device
    global a
    global h
    global spkrs

    a = arguments
    idx = queue.get()
    device = idx

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g, device="cpu")
    generator.load_state_dict(state_dict_g['generator'])
    

    file_list = parse_manifest(a.input_code_file)
    dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                            fmax_loss=h.fmax_for_loss, device=device,
                            f0=h.get('f0', None), f0_stats=a.f0_stats, f0_normalize=h.get('f0_normalize', False),
                            f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                            f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                            pad=a.pad, random_sample=h.get('random_sample', False),
                            rate=h.get('rate', False), boundary=h.get('boundary', False))

    os.makedirs(a.output_dir, exist_ok=True)

    if a.random_speakers:
        spkrs = random.sample(dataset.spkrs, k=min(a.n, len(dataset.spkrs))) # randomly choose a.n speakers to generate
    else:
        spkrs = dataset.spkrs # using all test speakers

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def inference(item_index):
    code, gt_audio, filename, _ = dataset[item_index]
    code_needs_grad = {}
    for k, v in code.items():
        if isinstance(v,np.ndarray): code_needs_grad[k] = torch.from_numpy(v).unsqueeze(0).to(device)
    code = code_needs_grad

    if a.dataset_type == "vctk":
        name = Path(filename).stem
        parts = name.split("_")
        fname_out_name = "_".join([parts[0], parts[1], parts[-1]])
    else:
        fname_out_name = Path(filename).stem 

    if h.get('f0_vq_params', None) or h.get('f0_quantizer', None):
        to_remove = gt_audio.shape[-1] % (16 * 80) 
        assert to_remove % h['code_hop_size'] == 0

        if to_remove != 0:
            to_remove_code = to_remove // h['code_hop_size']
            to_remove_f0 = to_remove // 80

            gt_audio = gt_audio[:-to_remove]
            code['code'] = code['code'][..., :-to_remove_code]
            code['f0'] = code['f0'][..., :-to_remove_f0]

            if h.get('rate', None):
                code['rate'] = code['rate'][..., :-to_remove_code]

    # cross-synthesis
    if h.spk_embed:
        embeds_all = np.load(a.spk_embed, allow_pickle=True)

    for _, k in enumerate(spkrs): # k is the reasinged spk id for TARGET speaker
        new_code = dict(code)
        if 'f0' in new_code:
            del new_code['f0']
            new_code['f0'] = code['f0']

        if 'spk_embed' in new_code:
            del new_code['spk_embed']
            embeds = []
            embeds += [torch.FloatTensor([float(x) for x in embeds_all[k]]).numpy()]
            new_code['spk_embed'] = torch.FloatTensor(np.array(embeds)).to(device)

        if a.f0_stats is not None:
            f0_stats_ = np.load(a.f0_stats, allow_pickle=True)

        if  h.get('f0', None) is not None and not h.get('f0_normalize', False):
            spkr = k
            f0 = code['f0'].clone()

            ii = (f0 != 0)
            mean_, std_ = f0[ii].mean(), f0[ii].std()
            if spkr not in f0_stats_:
                new_mean_, new_std_ = f0_stats_['f0_mean'], f0_stats_['f0_std']
            else:
                new_mean_, new_std_ = f0_stats_[spkr]['f0_mean'], f0_stats_[spkr]['f0_std']

            f0[ii] -= mean_
            f0[ii] /= std_
            # use the mean and std of target speaker to normalize f0
            f0[ii] *= new_std_
            f0[ii] += new_mean_
            new_code['f0'] = f0


        if h.get('f0_feats', False): # use the f0_feats of the target speaker
            if k not in f0_stats_:
                mean = f0_stats_['f0_mean']
                std = f0_stats_['f0_std']
            else:
                mean = f0_stats_[k]['f0_mean']
                std = f0_stats_[k]['f0_std']
            new_code['f0_stats'] = torch.FloatTensor([mean, std]).view(1, -1).to(device)

            output_file = os.path.join(a.output_dir, fname_out_name + f'_{k}.wav')

            audio, rtf = generate(h, generator, new_code)
            save_audio(output_file, audio, h.sampling_rate)

        #f0 control after speed control
        f0 = new_code['f0'].cpu()

        # find the first non-zero f0 and last non-zero f0
        start = 0
        end = 0
        for i, v in enumerate(f0[0, 0, :]):
            if torch.is_nonzero(v):
                start = i
                break
        for i in range(len(f0[0, 0, :]) - 1, 0, -1):
            if torch.is_nonzero(f0[0, 0, i]):
                end = i
                break
        
        mode = a.f0_curve_mode
        curve = gen_curve(end - start, mode)
        f0[0, 0, start:end] *= np.array(curve)
        new_code['f0'] = f0.to(device)
        
        output_file = os.path.join(a.output_dir, fname_out_name + f'_{k}_{mode}.wav')
        audio, rtf = generate(h, generator, new_code)
        save_audio(output_file, audio, h.sampling_rate)
            

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()  
    parser.add_argument('--input_code_file', default='./example/mani/test.txt') # input quantized code file
    parser.add_argument('--output_dir', default='generated_files') # where to store generated samples
    parser.add_argument('--checkpoint_file', required=True) # e.g.  checkpoints/vctk_huburt/g_00400000
    parser.add_argument('--f0_stats', type=Path) # f0 stats file
    parser.add_argument('--spk_embed', type=Path) # speaker embed file
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--dataset_type', default="vctk")
    parser.add_argument('--f0_curve_mode', default="stress")
    parser.add_argument('--random_speakers', action='store_true') # use n random speakers for conversion; \ 
                                                                  #  or use permutation of speakers in the datset
    parser.add_argument('-n', type=int, default=-1) # using n speakers in generation
    
    a = parser.parse_args()

    seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    # determine the output file names 
    model_id = Path(a.checkpoint_file).stem
    model_epoch = Path(cp_g).stem
    a.output_dir = os.path.join(a.output_dir, model_id, model_epoch)


    file_list = parse_manifest(a.input_code_file)
    dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                            f0=h.get('f0', None), f0_stats=a.f0_stats, f0_normalize=h.get('f0_normalize', False),
                            f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                            f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                            pad=a.pad, random_sample=h.get('random_sample', False),
                            rate=h.get('rate', False), boundary=h.get('boundary', False))
    if not a.multi_gpu:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i in range(0, len(dataset)):
            if a.n != -1 and i >= a.n:
                break
            inference(i)
            bar = progbar(i, len(dataset))
            message = f'{bar} {i}/{len(dataset)} '
            stream(message)
    else:
        idx = list(range(len(dataset)))
        random.shuffle(idx)
        with Pool(8, init_worker, (idQueue, a)) as pool:
            for i, _ in enumerate(pool.imap(inference, idx), 1):
                if a.n != -1 and i >= a.n:
                    break
                bar = progbar(i, len(idx))
                message = f'{bar} {i}/{len(idx)} '
                stream(message)

if __name__ == '__main__':
    main()
