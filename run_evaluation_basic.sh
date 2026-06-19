#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Define nodes and edges combinations
declare -a GRAPHS=("chain" "collider" "fork" "backdoor" "diamond")

for seed in {0..4}; do
    (
        for graph in "${GRAPHS[@]}"; do
            python evaluation_basic.py \
                --experimental_series paper_basic \
                --identifier "$graph" \
                --seed "$seed" \
                --num_samples 10000
        done
        echo "Finished all combinations for Seed $seed"
    )
done

wait 
echo "All seeds completed."
