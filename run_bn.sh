#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Define nodes and edges combinations
declare -a GRAPHS=("asia" "child" "alarm" "insurance" "win95pts" "hepar2" "hailfinder" "water" "barley" "mildew")


for seed in {0..4}; do
    (
        GPU=$((1 + seed % 5))
        for graph in "${GRAPHS[@]}"; do
            if ! CUDA_VISIBLE_DEVICES=$GPU python evaluation_bn.py \
                --experimental_series bn_for_workshop \
                --identifier "$graph" \
                --seed "$seed" \
                --num_samples 10000 \
                --max_epochs 20000; then
                echo "Run failed for graph $graph seed $seed" >&2
                continue
            fi
        done
        echo "Finished all combinations for Seed $seed"
    ) &
done

wait 
echo "All seeds completed."
