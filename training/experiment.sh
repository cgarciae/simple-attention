
# export CUDA_VISIBLE_DEVICES=""


python -m training.experiment \
    --data-path "data/spirals" \
    --batch-size 16 \
    --lr 0.001 \
    --epochs 200 \
    --n-layers 2 \
    --n-neurons 128 \
    "$@"