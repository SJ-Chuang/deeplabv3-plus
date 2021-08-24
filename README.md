# DeepLab V3+ for semantic segmentation

Semantic segmentation for reinforcement inspection

## Getting Started with DeepLabv3+

```bash
git clone https://github.com/SJ-Chuang/aisteel.git
cd aisteel
```

### Train

```bash
python3 deeplab.py \
    --image_path [path/to/image/directory] \
    --mask_path [path/to/mask/directory] \
    --save_dir [path/to/save/resurt] \
    --input_height 720 \
    --input_width 1280 \
    --input_channel 3 \
    --classes 2 \
    --activation softmax \
    --val_split 0.1 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --epochs 1000 \
```

### Test

```bash
python3 deeplab.py \
    --test_path [path/to/test_image/directory] \
    --model_path [path/to/model.h5] \
    --save_dir [path/to/save/resurt] \
    --input_height 720 \
    --input_width 1280 \
    --input_channel 3 \
    --classes 2 \
    --activation softmax \
```



