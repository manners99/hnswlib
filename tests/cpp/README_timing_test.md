# HNSW Degradation Test - Timing Only

This is a simplified version of the HNSW degradation test that focuses specifically on measuring:
1. **Iteration timing**: How long each deletion/insertion cycle takes
2. **Final recall**: The recall performance after all operations are complete

## Purpose

This test is designed for performance benchmarking where you want to:
- Measure the pure computational time of graph operations (without expensive graph analysis)
- Get the final recall score after the degradation process completes
- Avoid the overhead of detailed connectivity analysis during each iteration

## Usage

### Compilation
```bash
# Using make
make test_timing_only

# Or directly with g++
g++ -std=c++11 -O3 -march=native -DHAVE_CXX0X -fpic -ftree-vectorize degradation_test_timing_only.cpp -o degradation_test_timing_only
```

### Running the Test
```bash
./degradation_test_timing_only dimension initial_vectors deletion_batch insertion_batch num_iterations k M ef_construction lsh_threshold enable_lsh enable_replacement
```

### Parameters
- `dimension`: Vector dimension (128 for SIFT, 960 for GIST)
- `initial_vectors`: Number of vectors to start with
- `deletion_batch`: Number of vectors to delete per iteration
- `insertion_batch`: Number of vectors to insert per iteration
- `num_iterations`: Number of degradation iterations to run
- `k`: Number of nearest neighbors for recall testing
- `M`: HNSW M parameter
- `ef_construction`: HNSW ef_construction parameter
- `lsh_threshold`: LSH repair threshold (0.0 to disable LSH features)
- `enable_lsh`: 1 to enable LSH repair, 0 to disable
- `enable_replacement`: 1 to enable deleted point replacement, 0 for true degradation

### Example Commands

#### Vanilla HNSW without replacement (true degradation)
```bash
./degradation_test_timing_only 128 100000 10000 10000 100 10 16 200 0.0 0 0
```

#### HNSW with LSH repair and replacement
```bash
./degradation_test_timing_only 128 100000 10000 10000 100 10 16 200 0.15 1 1
```

#### Small test run for verification
```bash
./degradation_test_timing_only 128 1000 100 100 5 10 16 200 0.0 0 0
```

## Output Files

The test generates two CSV files:

### 1. Timing File (`*_timing_*.csv`)
Contains iteration-by-iteration timing data:
- `iteration`: Iteration number (0 = initial state)
- `iteration_time_seconds`: Time taken for this iteration's operations
- `cumulative_time_seconds`: Total elapsed time since test start

### 2. Final Recall File (`*_final_recall_*.csv`)
Contains final performance metrics:
- `final_recall`: Recall@k after all operations complete
- `final_search_time_ms`: Average search time in milliseconds
- `final_active_elements`: Number of active (non-deleted) elements
- `final_deleted_elements`: Number of deleted elements

## Key Differences from Full Degradation Test

**What's removed:**
- Graph connectivity analysis (BFS traversal, disconnected node counting)
- Per-iteration recall measurement
- Detailed connection statistics (inbound connections, connectivity density)
- LSH repair statistics tracking during iterations

**What's kept:**
- Precise timing of insertion/deletion operations
- Final recall measurement using ground truth
- All HNSW and LSH configuration options
- Real dataset loading (SIFT/GIST)

## Performance Benefits

This simplified test typically runs **5-10x faster** than the full degradation test for large datasets because:
- No expensive graph traversal during iterations
- No recall calculation during iterations (only at the end)
- Minimal metrics collection overhead
- Focus on core operation timing

## Use Cases

1. **Performance benchmarking**: Compare different HNSW configurations
2. **LSH repair evaluation**: Measure timing impact of LSH features
3. **Replacement strategy analysis**: Compare with/without deleted point replacement
4. **Large-scale testing**: Run longer tests without analysis overhead