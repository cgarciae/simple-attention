
# export CUDA_VISIBLE_DEVICES=""


python -m training.experiment \
    --data-path "data/spirals" \
    --n-classes 2 \
    --batch-size 16 \
    --lr 0.001 \
    --epochs 200 \
    --n-layers 3 \
    --n-neurons 128 \
    "$@"