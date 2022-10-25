"""
Add existing speaker embeding to pasered code file.
"""

import argparse
from pathlib import Path
import numpy as np

def parse_manifest(manifest):
    audio_files = []
    codes = []
    spk_names = []
    durations = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                codes += [sample['hubert']]
                audio_files += [sample["audio"]]
                durations += [sample["duration"]]
                spk_names += [Path(sample["audio"]).stem.split("_")[0]]
            else:
                audio_files += [line.strip()]
                durations += [sample["duration"]]
                spk_names += [Path(sample["audio"]).stem.split("_")[0]]

    return audio_files, codes, durations, spk_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_file', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    

    audio_files, codes, durations, spk_names = parse_manifest(args.manifest)
    embeds = np.load(args.embed_file, allow_pickle=True)

    lines = []
    for i, (audio_path, code, duration, spk_name) in enumerate(zip(audio_files, codes, durations, spk_names)):
        line = {}
        line["audio"] = audio_path
        line["hubert"] = code
        line["duration"] = duration
        line["spk_embed"] = ' '.join([str(elem) for elem in embeds[spk_name]])

        lines += [line]
    
    args.outdir.mkdir(exist_ok=True, parents=True)
    outfile = args.outdir / args.manifest.name
    with open(outfile, 'w') as f:
        f.write('\n'.join([str(x) for x in lines]))

    print("Parsed file was saved to ", str(outfile))

if __name__ == "__main__":
    main()

    