
# export CUDA_VISIBLE_DEVICES=""


python -m training.experiment \
    --data-path "data/spirals" \
    --n-classes 2 \
    --batch-size 8 \
    --lr 0.0002 \
    --epochs 500 \
    --n-layers 2 \
    --n-neurons 64 \
    "$@"