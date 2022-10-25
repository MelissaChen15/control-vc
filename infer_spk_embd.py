"""
Generate speaker embeddings and metadata for training

Reference:
    https://github.com/auspicious3000/autovc/blob/master/make_metadata.py
"""
import os
from models import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import pickle
import argparse
from pathlib import Path

from tqdm import tqdm

def infer(in_dir, out_dir, ckpt_path, num_utts=10, len_crop=128):
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    
    sub_dirs = list(in_dir.glob("*"))

    speakers = {}
    print('Processing speakers:')
    for sub_dir in tqdm(sorted(sub_dirs), total=len(sub_dirs)):
        speaker =  sub_dir.name
        filenames = list(sub_dir.glob("*"))

        if num_utts == -1: # use all utterance under the folder
            num_utts_local = len(filenames)
        if len_crop == -1:
            lens = [np.load(f).shape[0] for f in filenames]
            len_crop_local = min(lens) - 1
        
        # make speaker embedding
        assert len(filenames) >= num_utts_local
        idx = np.random.choice(len(filenames), size=num_utts_local, replace=False)
        embs = []
        for i in range(num_utts_local):
            tmp = np.load(filenames[idx[i]])
            candidates = np.delete(np.arange(len(filenames)), idx)
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop_local:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(root_dir, speaker, filenames[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
            left = np.random.randint(0, tmp.shape[0]-len_crop_local)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop_local, :]).cuda()
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())     

        # speakers.append(utterances)
        speakers[speaker] = np.mean(embs, axis=0)
    
    outname = os.path.join(out_dir, 'spk_embed.pkl')
    with open(outname, "wb") as f:
        pickle.dump(speakers, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--num_utts', type=int, required=False)
    parser.add_argument('--len_crop', type=int, required=False)
    args = parser.parse_args()

    infer(args.srcdir, args.outdir, args.checkpoint_path, args.num_utts, args.len_crop)

if __name__ == "__main__":
    main()