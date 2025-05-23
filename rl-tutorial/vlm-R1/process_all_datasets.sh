#!/bin/bash

# Dataset Processing Script Executor
# Run in vlm-R1 directory to process all math datasets

echo "Starting to process all datasets..."
echo "Current working directory: $(pwd)"

# Create data output directory
mkdir -p ./data

# Set common parameters
LOCAL_DIR="./data"
# Define datasets and their corresponding Python modules
declare -A DATASETS=(
    ["We-Math"]="process_we_math"
    ["PolyMath"]="process_polymath"
    ["MathVision"]="process_math_vision"
    ["Geo3K"]="process_geo3k"
    ["Skywork"]="process_skywork"
    ["AIME24"]="process_aime24"
    ["Math500"]="process_math500"
)
# Loop through datasets and process each one
for DATASET in "${!DATASETS[@]}"; do
    MODULE=${DATASETS[$DATASET]}
    echo "========================================"
    echo "Processing $DATASET dataset..."
    echo "========================================"
    python -m "data_processor.$MODULE" --local_dir "$LOCAL_DIR"
    if [ $? -eq 0 ]; then
        echo "✓ $DATASET dataset processing completed"
    else
        echo "✗ $DATASET dataset processing failed"
    fi
done

echo "========================================"
echo "All datasets processing completed!"
echo "Processed data saved in: $LOCAL_DIR"
echo "========================================"

# Display processing results statistics
echo "Processing Results Statistics:"
ls -la "$LOCAL_DIR"/*.parquet 2>/dev/null | wc -l | xargs -I {} echo "Number of successfully processed datasets: {}"
echo "Data files list:"
ls -la "$LOCAL_DIR"/*.parquet 2>/dev/null || echo "No .parquet files found" 