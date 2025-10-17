#!/bin/bash

# Script to download SIFT dataset for testing
# This downloads a smaller subset suitable for testing

echo "SIFT Dataset Download Script"
echo "============================"

# Create directory structure
mkdir -p bigann/gnd

echo "Note: This script downloads the smaller SIFT datasets suitable for testing."
echo "For the full SIFT-1B dataset, visit: http://corpus-texmex.irisa.fr/"
echo ""

# Download SIFT10K (smaller dataset for quick testing)
echo "Downloading SIFT10K base vectors..."
if [ ! -f "sift_base.fvecs" ]; then
    wget -q --show-progress http://corpus-texmex.irisa.fr/texmex/corpus/sift.tar.gz
    tar -xzf sift.tar.gz
    echo "SIFT10K base vectors downloaded (sift_base.fvecs)"
else
    echo "sift_base.fvecs already exists"
fi

echo "Downloading SIFT10K query vectors..."
if [ ! -f "sift_query.fvecs" ]; then
    # Query vectors are in the same archive
    echo "sift_query.fvecs extracted"
else 
    echo "sift_query.fvecs already exists"
fi

echo "Downloading SIFT10K ground truth..."
if [ ! -f "sift_groundtruth.ivecs" ]; then
    # Ground truth is in the same archive  
    echo "sift_groundtruth.ivecs extracted"
else
    echo "sift_groundtruth.ivecs already exists"
fi

# For SIFT-1B (commented out due to large size)
echo ""
echo "For SIFT-1B dataset (warning: very large files!):"
echo "  Base vectors (~120GB): wget http://corpus-texmex.irisa.fr/texmex/corpus/bigann_base.bvecs"
echo "  Query vectors (~50MB): wget http://corpus-texmex.irisa.fr/texmex/corpus/bigann_query.bvecs"
echo "  Ground truth (varies): wget http://corpus-texmex.irisa.fr/texmex/corpus/bigann_gnd.tar.gz"

echo ""
echo "To use with the SIFT degradation test:"
echo "  1. For SIFT10K: Convert .fvecs to .bvecs format (or modify test to handle .fvecs)"
echo "  2. For SIFT-1B: Download the .bvecs files to bigann/ directory"
echo ""
echo "Test can be run with:"
echo "  make test_degradation_sift"
echo "  # or"
echo "  ./tests/cpp/sift_degradation_test [base_file] [query_file] [gt_file]"
