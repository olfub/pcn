#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Problem sizes (number of variables) to benchmark.
declare -a SIZES=(2 4 8 16 32 64 128)

# Keep these defaults aligned with evaluation_ipf_random.py.
CARDINALITY=4
EPSILON=1e-6
MAX_ITERATIONS=1000
DEVICE=cuda

for seed in {0..1}; do
    (
        # GPU=$((seed % 5))
        # GPU=$((GPU >= 4 ? GPU + 1 : GPU))
        GPU=$((6 + seed % 2))

        for size in "${SIZES[@]}"; do
            OUTPUT_DIR="experiments/ipf_random/size_${size}/seed_${seed}"

            if ! CUDA_VISIBLE_DEVICES=$GPU python evaluation_ipf_random.py \
                --size "$size" \
                --cardinality "$CARDINALITY" \
                --epsilon "$EPSILON" \
                --max-iterations "$MAX_ITERATIONS" \
                --seed "$seed" \
                --device "$DEVICE" \
                --output-dir "$OUTPUT_DIR"; then
                echo "Run failed for size $size seed $seed" >&2
                continue
            fi
        done

        echo "Finished all combinations for Seed $seed"
    ) &
done

wait
echo "All seeds completed."