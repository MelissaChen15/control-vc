CUDA_VISIBLE_DEVICES=0
BASE_DIR=./example
CKPT_DIR=./checkpoints
MANI_DIR=${BASE_DIR}/mani
OUT_DIR=${BASE_DIR}/synthesis
WAV_DIR_IN=${BASE_DIR}/inputs
WAV_DIR_PROCESSED=${MANI_DIR}/wavs_padded
EXT=.flac
OUT_QUANTIZED_FILE=${MANI_DIR}/hubert_codes.txt
MEL_DIR=$MANI_DIR/mels

# stage 1: preprocess -  trim, pad and sample rate conversion
echo "=========== Preprocessing... ==========="
python3 scripts/preprocess.py \
    --srcdir $WAV_DIR_IN \
    --outdir $WAV_DIR_PROCESSED \
    --postfix $EXT \
    --pad --keepfolder \
    --rhythm_cruve

# stage 2: get quantized hubert code
echo "=========== Extracting and parsing HuBERT code... ==========="
python3 infer_hubert.py \
    --feature_type hubert \
    --kmeans_model_path ${CKPT_DIR}/km.bin \
    --acoustic_model_path ${CKPT_DIR}/hubert_base_ls960.pt \
    --layer 6 \
    --wav_path $WAV_DIR_PROCESSED \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXT

python3 scripts/parse_hubert_codes.py \
    --codes $OUT_QUANTIZED_FILE \
    --manifest ${MANI_DIR}/wavlist.txt \
    --outdir $MANI_DIR \
    --all-test

# stage 3: get speaker embedding
echo "=========== Extracting speaker embedding... ==========="
python3 scripts/extract_mel4spkembd.py \
    --wavdir $WAV_DIR_PROCESSED \
    --meldir $MEL_DIR \
    --ext $EXT

python3 infer_spk_embd.py \
    --srcdir $MEL_DIR \
    --outdir $MANI_DIR \
    --checkpoint_path ${CKPT_DIR}/3000000-BL.ckpt \
    --num_utts -1 \
    --len_crop -1

python scripts/parse_spk_embed.py \
    --embed_file ${MANI_DIR}/spk_embed.pkl \
    --manifest ${MANI_DIR}/test.txt \
    --outdir $MANI_DIR

python scripts/get_f0_stats.py \
    --srcdir $WAV_DIR_PROCESSED \
    --outdir $MANI_DIR

# stage 4: generate audio files
echo "=========== Generating audio... ==========="
python infer_main.py \
     --input_code_file ${MANI_DIR}/test.txt \
     --checkpoint_file ${CKPT_DIR}/embed_f0stat2 \
     --output_dir $OUT_DIR \
     --f0_stats ${MANI_DIR}/f0_stats.pkl \
     --spk_embed ${MANI_DIR}/spk_embed.pkl 