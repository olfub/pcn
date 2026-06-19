#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

declare -a GRID_PARAMS=(4 5 6 7 8)

for seed in {0..4}; do
    (
        GPU=$((1+ seed % 5))
        # GPU=$((GPU >= 4 ? GPU + 1 : GPU))
        for grid_param in "${GRID_PARAMS[@]}"; do
            if ! CUDA_VISIBLE_DEVICES=$GPU python evaluation_worst_case.py \
                --experimental_series bn_for_workshop_grids \
                --graph_type grid \
                --graph_param "$grid_param" \
                --seed "$seed" \
                --num_samples 10000 \
                --max_epochs 20000; then
                echo "Run failed for grid_param $grid_param seed $seed" >&2
                continue
            fi
        done
        echo "Finished all grid parameters for Seed $seed"
    ) &
done

wait
echo "All seeds completed."
