"""
Get quantized HuBERT from pretrained models  

Reference:
https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit
"""

import argparse
import logging
import os
from pathlib import Path
import joblib
from typing import List, Tuple

import numpy as np

from fairseq_feature_reader import get_features
from utils import get_wavlist4hubert
import soundfile as sf


def get_audio_files(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                len(items) == 2
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--wav_path",
        type=Path,
        default=None,
        help="Root dir containing wav files",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    return parser


def main(args, logger):
    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Geting manifest...")
        manifest_path = args.wav_path.parent / "wavlist.txt"
        get_wavlist4hubert(args.wav_path, manifest_path, args.extension)
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        features_batch = get_features(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=manifest_path,
            sample_pct=1.0,
            flatten=False,
        )
        logger.info(
            f"Features extracted for {len(features_batch)} utterances.\n"
        )
        logger.info(
            f"Dimensionality of representation = {features_batch[0].shape[1]}"
        )

    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    _, fnames, _ = get_audio_files(manifest_path)

    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    print(f"Writing quantized predictions to {args.out_quantized_file_path}")
    with open(args.out_quantized_file_path, "w") as fout:
        for i, feats in enumerate(features_batch):
            pred = kmeans_model.predict(feats)
            pred_str = " ".join(str(p) for p in pred)
            base_fname = os.path.basename(fnames[i]).rstrip(args.extension)
            fout.write(f"{base_fname}|{pred_str}\n")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
