# MM-Fundus-CLIP
MM-Fundus-CLIP: Development of a Multi-Modality Fundus Foundation Model Leveraging Large Language Model and Contrastive Language-Image Pre-Training

## Running a pre-tuning recipe
```
python

import os
GPU='0'
image_dir='./datasets/'
train_data='./datasets/train.csv'
val_data='./datasets/val.csv'
logs_dir='./results_clip'
name='uwf_clip'
cache_dir='./models/clip'
cmd=f'PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python open_clip_train/main.py \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data {train_data} \
    --val-data {val_data} \
    --image-dir {image_dir} \
    --dataset-type csv \
    --csv-separator "," \
    --csv-img-key image \
    --csv-caption-key text \
    --extra-aug \
    --lr 1e-6 \
    --beta1 0.9 \
    --beta2 0.95 \
    --lr-scheduler const \
    --warmup 0 \
    --wd 0.2 \
    --batch-size 128 \
    --epochs 10 \
    --workers 8 \
    --model "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384" \
    --cache-dir {cache_dir} \
    --precision amp_bf16 \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logs {logs_dir} \
    --name {name}'
print(cmd)
os.system(cmd)
```

## Running a zero-shot classification
```
python main_clip_zero.py
```

## Requirements
```
pip install open_clip_torch==2.30.0
pip install open_clip_torch[training]==2.30.0
pip uninstall open_clip_torch -y
```

## Acknowledgements
```
https://github.com/mlfoundations/open_clip/tree/v2.30.0
https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
```
