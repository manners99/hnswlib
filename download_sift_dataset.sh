#!/bin/bash

echo "SIFT Dataset Downloader"
echo "======================="
echo ""

# Function to download and check file
download_file() {
    local url=$1
    local filename=$2
    local size_mb=$3
    
    if [ -f "$filename" ]; then
        echo "✓ $filename already exists"
        return 0
    fi
    
    echo "Downloading $filename (~${size_mb}MB)..."
    wget -c --progress=bar:force:noscroll "$url" -O "$filename"
    
    if [ $? -eq 0 ]; then
        echo "✓ Downloaded $filename successfully"
        return 0
    else
        echo "✗ Failed to download $filename"
        rm -f "$filename"
        return 1
    fi
}

# Create directory structure
mkdir -p bigann/gnd

echo "This script downloads SIFT datasets for testing."
echo "Choose your dataset size:"
echo ""
echo "1) SIFT 10K   - Small dataset for quick testing (~13MB total)"
echo "2) SIFT 100K  - Medium dataset for thorough testing (~130MB total)"  
echo "3) SIFT 1M    - Large dataset subset (~1.3GB total)"
echo "4) SIFT 1B    - Full billion-scale dataset (~128GB+ total - WARNING: VERY LARGE!)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Downloading SIFT 10K dataset..."
        cd tests/cpp
        download_file "http://corpus-texmex.irisa.fr/texmex/corpus/sift.tar.gz" "sift.tar.gz" "13"
        
        if [ $? -eq 0 ]; then
            echo "Extracting..."
            tar -xzf sift.tar.gz
            echo "✓ SIFT 10K dataset ready"
            echo ""
            echo "Files available:"
            ls -lh sift_*.fvecs sift_*.ivecs 2>/dev/null
            echo ""
            echo "Note: These are .fvecs/.ivecs files. For .bvecs format, use SIFT 1B dataset."
        fi
        ;;
        
    2)
        echo "SIFT 100K is not directly available. Using SIFT 10K instead."
        echo "You can truncate SIFT 1M to get 100K vectors if needed."
        ;;
        
    3)
        echo "Downloading SIFT 1M subset..."
        cd bigann
        
        # Download 1M subset (this is manageable)
        download_file "http://corpus-texmex.irisa.fr/texmex/corpus/sift1M.tar.gz" "sift1M.tar.gz" "1300"
        
        if [ $? -eq 0 ]; then
            echo "Extracting..."
            tar -xzf sift1M.tar.gz
            echo "✓ SIFT 1M dataset ready"
        fi
        ;;
        
    4)
        echo "WARNING: SIFT 1B dataset is extremely large!"
        echo "Base vectors: ~120GB"
        echo "Query vectors: ~50MB" 
        echo "Ground truth: varies"
        echo ""
        read -p "Are you sure you want to continue? (y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            cd bigann
            
            echo "Downloading query vectors first (small)..."
            download_file "http://corpus-texmex.irisa.fr/texmex/corpus/bigann_query.bvecs" "bigann_query.bvecs" "50"
            
            echo ""
            echo "Downloading ground truth..."
            download_file "http://corpus-texmex.irisa.fr/texmex/corpus/bigann_gnd.tar.gz" "bigann_gnd.tar.gz" "varies"
            
            if [ -f "bigann_gnd.tar.gz" ]; then
                echo "Extracting ground truth..."
                tar -xzf bigann_gnd.tar.gz
            fi
            
            echo ""
            echo "For base vectors (~120GB), run manually:"
            echo "  cd bigann"
            echo "  wget http://corpus-texmex.irisa.fr/texmex/corpus/bigann_base.bvecs"
            echo ""
            echo "Or download in chunks if connection is unstable."
        else
            echo "Cancelled SIFT 1B download."
        fi
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Dataset download completed!"
echo ""
echo "To run the large-scale test:"
echo "  make test_large_sift"
echo ""
echo "Or with specific files:"
echo "  ./tests/cpp/large_sift_degradation_test [base_file] [query_file] [gt_file]"
echo ""
echo "For synthetic data (no download needed):"
echo "  ./tests/cpp/large_sift_degradation_test"
