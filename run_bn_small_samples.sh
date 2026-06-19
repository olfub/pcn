#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Focused small-sample evaluation set.
declare -a GRAPHS=("asia" "child" "insurance" "alarm")
declare -a SAMPLE_SIZES=(50 100 250 500 1000 10000)

for seed in {0..4}; do
    (
        GPU=$((1 + seed % 5))

        for graph in "${GRAPHS[@]}"; do
            for num_samples in "${SAMPLE_SIZES[@]}"; do
                if ! CUDA_VISIBLE_DEVICES=$GPU python evaluation_bn_pcn_einet.py \
                    --experimental_series bn_for_workshop_samples \
                    --identifier "$graph" \
                    --seed "$seed" \
                    --num_samples "$num_samples" \
                    --max_epochs 20000; then
                    echo "Run failed for graph $graph seed $seed num_samples $num_samples" >&2
                    continue
                fi
            done
        done

        echo "Finished all combinations for Seed $seed"
    ) &
done

wait
echo "All seeds completed."
