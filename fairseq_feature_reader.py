"""
Feature reader for pre-trained hubert model

Reference:
 https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit
"""

import gc
import os
import random
import shutil
import numpy as np

import torch
import tqdm

import fairseq
import soundfile as sf
import torch.nn.functional as F


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None):
        x = self.read_audio(file_path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def get_feature_reader(feature_type):
    if feature_type == "hubert":
        return HubertFeatureReader
    else:
        raise NotImplementedError(f"{feature_type} is not supported.")


def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct
):  
    feature_reader_cls = get_feature_reader(feature_type)
    with open(manifest_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        file_path_list = [
            os.path.join(root, line.split("\t")[0])
            for line in lines
            if len(line) > 0
        ]
        if sample_pct < 1.0:
            file_path_list = random.sample(
                file_path_list, int(sample_pct * len(file_path_list))
            )
        num_files = len(file_path_list)
        reader = feature_reader_cls(
            checkpoint_path=checkpoint_path, layer=layer
        )

        def iterate():
            for file_path in file_path_list:
                feats = reader.get_feats(file_path)
                yield feats.cpu().numpy()

    return iterate, num_files


def get_features(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
    )
    iterator = generator()

    features_list = []
    for features in tqdm.tqdm(iterator, total=num_files):
        features_list.append(features)

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if flatten:
        return np.concatenate(features_list)

    return features_list


def get_and_dump_features(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    flatten,
    out_features_path,
):  
    # Feature extraction
    features_batch = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        flatten=flatten,
    )

    # Save features
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, features_batch)

    return features_batch
