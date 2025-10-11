#include "../../hnswlib/hnswlib.h"
#include "fvecs_loader.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <queue>

// Forward declaration
std::unordered_set<hnswlib::tableint> findReachableNodes(hnswlib::HierarchicalNSW<float>* index);

// # Vanilla HNSW with replacement
// ./degradation_test_with_output 128 100000 10000 10000 100 100 32 200 0.0 0 1

// # LSH repair without replacement  
// ./degradation_test_with_output 128 100000 10000 10000 100 100 32 200 0.15 1 0

// # LSH repair with replacement
// ./degradation_test_with_output 128 100000 10000 10000 100 100 32 200 0.15 1 1

// # Vanilla HNSW without replacement (true degradation)
// ./degradation_test_with_output 128 100000 10000 10000 100 100 32 200 0.0 0 0

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }
    
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    
    void reset() { time_begin = std::chrono::steady_clock::now(); }
};

struct TestMetrics {
    int iteration;
    int active_elements;
    int deleted_elements;
    double recall;
    double search_time_ms;
    int total_connections;
    int disconnected_nodes;
    int unreachable_deleted_nodes;  // New: count of deleted nodes that were unreachable from entry point
    double avg_inbound_connections;
    int min_inbound_connections;
    int max_inbound_connections;
    double connectivity_density;
    double iteration_time_seconds;
    double cumulative_time_seconds;
    
    // LSH repair statistics
    long lsh_repair_calls;
    long lsh_repairs_performed;
    long lsh_queries_made;
    long lsh_connections_added;
    
    void writeCSVHeader(std::ofstream& file) {
        file << "iteration,active_elements,deleted_elements,recall,search_time_ms,"
             << "total_connections,disconnected_nodes,unreachable_deleted_this_iter,avg_inbound,min_inbound,max_inbound,"
             << "connectivity_density,iteration_time_seconds,cumulative_time_seconds,"
             << "lsh_repair_calls,lsh_repairs_performed,lsh_queries_made,lsh_connections_added\n";
    }
    
    void writeCSVRow(std::ofstream& file) {
        file << iteration << "," << active_elements << "," << deleted_elements << ","
             << std::fixed << std::setprecision(4) << recall << ","
             << std::setprecision(2) << search_time_ms << ","
             << total_connections << "," << disconnected_nodes << "," << unreachable_deleted_nodes << ","
             << std::setprecision(2) << avg_inbound_connections << ","
             << min_inbound_connections << "," << max_inbound_connections << ","
             << std::setprecision(4) << connectivity_density << ","
             << std::setprecision(2) << iteration_time_seconds << ","
             << std::setprecision(2) << cumulative_time_seconds << ","
             << lsh_repair_calls << "," << lsh_repairs_performed << ","
             << lsh_queries_made << "," << lsh_connections_added << "\n";
    }
    
    void printConsole() {
        std::cout << std::setw(4) << iteration 
                  << std::setw(8) << active_elements 
                  << std::setw(8) << deleted_elements
                  << std::setw(8) << std::fixed << std::setprecision(3) << recall
                  << std::setw(10) << std::setprecision(1) << search_time_ms
                  << std::setw(8) << total_connections
                  << std::setw(6) << disconnected_nodes
                  << std::setw(7) << unreachable_deleted_nodes
                  << std::setw(8) << std::setprecision(1) << avg_inbound_connections
                  << std::setw(6) << lsh_repairs_performed
                  << std::setw(6) << lsh_connections_added
                  << std::setw(8) << std::setprecision(1) << iteration_time_seconds
                  << std::endl;
    }
    
    static void printConsoleHeader() {
        std::cout << std::setw(4) << "Iter"
                  << std::setw(8) << "Active" 
                  << std::setw(8) << "Deleted"
                  << std::setw(8) << "Recall"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(8) << "Conns"
                  << std::setw(6) << "Disco"
                  << std::setw(7) << "UnrDel"
                  << std::setw(8) << "AvgIn"
                  << std::setw(6) << "Repair"
                  << std::setw(6) << "NewCon"
                  << std::setw(8) << "IterSec"
                  << std::endl;
    }
};

// Generate a unique filename with timestamp
std::string generateFilename(const std::string& base, const std::string& ext) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream oss;
    oss << base << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << "." << ext;
    return oss.str();
}

TestMetrics analyzeGraph(hnswlib::HierarchicalNSW<float>* index, 
                        hnswlib::BruteforceSearch<float>* brute_force,
                        const std::vector<float>& base_data,
                        const std::vector<float>& query_data,
                        int dimension, int k, 
                        StopW& iteration_timer, StopW& total_timer,
                        int iteration) {
    TestMetrics metrics;
    metrics.iteration = iteration;
    
    // Use provided query data (either real dataset queries or pre-generated synthetic ones)
    
    // Measure search performance with proper ground truth from brute force
    int total_queries = 100;
    int total_correct = 0;
    int total_results = 0;
    double total_hnsw_search_time_micro = 0.0;
    
    for (int i = 0; i < total_queries; i++) {
        // Get ground truth from brute force (not timed)
        auto gt_result = brute_force->searchKnn(query_data.data() + i * dimension, k);
        std::unordered_set<hnswlib::labeltype> gt_set;
        while (!gt_result.empty()) {
            gt_set.insert(gt_result.top().second);
            gt_result.pop();
        }
        
        // Time only the HNSW search
        StopW hnsw_search_timer;
        auto hnsw_result = index->searchKnn(query_data.data() + i * dimension, k);
        total_hnsw_search_time_micro += hnsw_search_timer.getElapsedTimeMicro();
        
        // Count correct results (not timed)
        while (!hnsw_result.empty()) {
            if (gt_set.find(hnsw_result.top().second) != gt_set.end()) {
                total_correct++;
            }
            hnsw_result.pop();
            total_results++;
        }
    }
    
    metrics.recall = total_results > 0 ? static_cast<double>(total_correct) / total_results : 0.0;
    metrics.search_time_ms = total_hnsw_search_time_micro / 1000.0;
    
    // Analyze graph structure
    int cur_element_count = index->cur_element_count;
    metrics.active_elements = cur_element_count - index->getDeletedCount();
    metrics.deleted_elements = index->getDeletedCount();
    
    // Find all nodes reachable from entry point using BFS
    std::unordered_set<hnswlib::tableint> reachable_nodes = findReachableNodes(index);
    
    std::vector<int> inbound_connections(cur_element_count, 0);
    int total_connections = 0;
    int disconnected = 0;
    
    // Count all connections and calculate inbound connections for statistics
    for (int i = 0; i < cur_element_count; i++) {
        if (index->isMarkedDeleted(i)) continue;
        
        for (int l = 0; l <= index->element_levels_[i]; l++) {
            auto ll_cur = index->get_linklist_at_level(i, l);
            int size = index->getListCount(ll_cur);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
            
            total_connections += size;
            
            for (int j = 0; j < size; j++) {
                if (data[j] < cur_element_count && !index->isMarkedDeleted(data[j])) {
                    inbound_connections[data[j]]++;
                }
            }
        }
    }
    
    // Calculate connection statistics and count unreachable nodes
    std::vector<int> valid_inbound;
    for (int i = 0; i < cur_element_count; i++) {
        if (!index->isMarkedDeleted(i)) {
            valid_inbound.push_back(inbound_connections[i]);
            
            // Count as disconnected if not reachable from entry point via BFS
            if (reachable_nodes.find(i) == reachable_nodes.end()) {
                disconnected++;
            }
        }
    }
    
    metrics.total_connections = total_connections;
    metrics.disconnected_nodes = disconnected;
    
    // Note: unreachable_deleted_nodes is set externally to track cumulative count
    // of nodes that were already unreachable from entry point when they were deleted
    
    if (!valid_inbound.empty()) {
        metrics.min_inbound_connections = *std::min_element(valid_inbound.begin(), valid_inbound.end());
        metrics.max_inbound_connections = *std::max_element(valid_inbound.begin(), valid_inbound.end());
        metrics.avg_inbound_connections = static_cast<double>(std::accumulate(valid_inbound.begin(), valid_inbound.end(), 0)) / valid_inbound.size();
        metrics.connectivity_density = static_cast<double>(total_connections) / metrics.active_elements;
    } else {
        metrics.min_inbound_connections = metrics.max_inbound_connections = 0;
        metrics.avg_inbound_connections = metrics.connectivity_density = 0;
    }
    
    // Timing information
    metrics.iteration_time_seconds = iteration_timer.getElapsedTimeMicro() / 1000000.0;
    metrics.cumulative_time_seconds = total_timer.getElapsedTimeMicro() / 1000000.0;
    
    // LSH repair statistics
    metrics.lsh_repair_calls = index->getLSHRepairCalls();
    metrics.lsh_repairs_performed = index->getLSHRepairsPerformed();
    metrics.lsh_queries_made = index->getLSHQueriesMade();
    metrics.lsh_connections_added = index->getLSHConnectionsAdded();
    
    return metrics;
}

// BFS to find all nodes reachable from entry point
std::unordered_set<hnswlib::tableint> findReachableNodes(hnswlib::HierarchicalNSW<float>* index) {
    std::unordered_set<hnswlib::tableint> reachable;
    std::unordered_set<hnswlib::tableint> visited;  // Track all visited nodes (including deleted)
    
    // If no elements or no entry point, return empty set
    if (index->cur_element_count == 0 || index->getEntryPoint() == -1) {
        return reachable;
    }
    
    std::queue<hnswlib::tableint> bfs_queue;
    hnswlib::tableint entry_point = index->getEntryPoint();
    
    // Start BFS from entry point
    bfs_queue.push(entry_point);
    visited.insert(entry_point);
    
    // Only count entry point as reachable if it's not deleted
    if (!index->isMarkedDeleted(entry_point)) {
        reachable.insert(entry_point);
    }
    
    while (!bfs_queue.empty()) {
        hnswlib::tableint current = bfs_queue.front();
        bfs_queue.pop();
        
        // Only check layer 0 connections for reachability
        auto ll_cur = index->get_linklist_at_level(current, 0);
        int size = index->getListCount(ll_cur);
        hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
        
        for (int i = 0; i < size; i++) {
            hnswlib::tableint neighbor = data[i];
            
            // Visit all valid neighbors we haven't seen before (including deleted ones)
            if (neighbor < index->cur_element_count && 
                visited.find(neighbor) == visited.end()) {
                
                visited.insert(neighbor);
                bfs_queue.push(neighbor);
                
                // Only count as reachable if the neighbor is NOT deleted
                if (!index->isMarkedDeleted(neighbor)) {
                    reachable.insert(neighbor);
                }
            }
        }
    }
    
    return reachable;
}

int main(int argc, char* argv[]) {
    std::cout << "HNSW Degradation Test with Data Logging" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Test parameters - configurable via command line
    int dimension = 128;              // Default SIFT dimension (use 960 for GIST)
    int initial_vectors = 100000;     // Default 100K
    int deletion_batch_size = 10000;  // Default 10K
    int insertion_batch_size = 10000; // Default 10K
    int num_iterations = 100;         // Default 100
    int k = 10;                       // Default k for recall testing
    int M = 16;                       // Default HNSW M parameter
    int ef_construction = 200;        // Default HNSW ef_construction
    double lsh_repair_threshold = 0.1; // Default LSH repair threshold
    bool enable_lsh_repair = true;    // Default LSH repair enabled
    bool enable_replacement = false;  // Default replacement disabled
    
    // Parse command line arguments
    if (argc >= 2) dimension = std::atoi(argv[1]);
    if (argc >= 3) initial_vectors = std::atoi(argv[2]);
    if (argc >= 4) deletion_batch_size = std::atoi(argv[3]);
    if (argc >= 5) insertion_batch_size = std::atoi(argv[4]);
    if (argc >= 6) num_iterations = std::atoi(argv[5]);
    if (argc >= 7) k = std::atoi(argv[6]);
    if (argc >= 8) M = std::atoi(argv[7]);
    if (argc >= 9) ef_construction = std::atoi(argv[8]);
    if (argc >= 10) lsh_repair_threshold = std::atof(argv[9]);
    if (argc >= 11) enable_lsh_repair = (std::atoi(argv[10]) != 0);
    if (argc >= 12) enable_replacement = (std::atoi(argv[11]) != 0);
    
    // Determine dataset type based on dimension
    std::string dataset_type = (dimension == 128) ? "SIFT" : "GIST";
    std::string dataset_path;
    
    if (dataset_type == "SIFT") {
        dataset_path = "../../datasets/sift/sift_base.fvecs";
    } else {
        dataset_path = "../../datasets/gist/gist_base.fvecs";
    }
    
    // LSH configuration
    int lsh_num_tables = 8;
    int lsh_num_hashes = 10;
    
    // Generate output filenames based on LSH configuration and dataset
    std::string base_filename;
    if (enable_lsh_repair) {
        base_filename = "degradation_test_" + dataset_type + "_lsh_" + std::to_string(lsh_num_tables) + "x" + std::to_string(lsh_num_hashes);
    } else {
        base_filename = "degradation_test_" + dataset_type + "_vanilla_hnsw";
    }
    std::string csv_filename = generateFilename(base_filename, "csv");
    std::string log_filename = generateFilename(base_filename, "log");
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dataset: " << dataset_type << " (real data)" << std::endl;
    std::cout << "  Test Type: " << (enable_replacement ? "REPLACEMENT MODE" : "TRUE DEGRADATION (no node replacement)") << std::endl;
    std::cout << "  Dimension: " << dimension << std::endl;
    std::cout << "  Initial vectors: " << initial_vectors << std::endl;
    std::cout << "  Deletion batch: " << deletion_batch_size << std::endl;
    std::cout << "  Insertion batch: " << insertion_batch_size << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  k (recall test): " << k << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW ef_construction: " << ef_construction << std::endl;
    std::cout << "  HNSW ef_search: " << std::max(400, initial_vectors / 100) << std::endl;
    std::cout << "  LSH repair enabled: " << (enable_lsh_repair ? "Yes" : "No") << std::endl;
    std::cout << "  Replacement enabled: " << (enable_replacement ? "Yes" : "No") << std::endl;
    if (enable_lsh_repair) {
        std::cout << "  LSH tables: " << lsh_num_tables << std::endl;
        std::cout << "  LSH hashes per table: " << lsh_num_hashes << std::endl;
        std::cout << "  LSH repair threshold: " << lsh_repair_threshold << std::endl;
    }
    std::cout << "  Output CSV: " << csv_filename << std::endl;
    std::cout << "  Output log: " << log_filename << std::endl;
    std::cout << std::endl;
    
    // Calculate total vectors needed: initial + (iterations * insertion_batch_size)
    int total_vectors_needed = initial_vectors + (num_iterations * insertion_batch_size);
    std::cout << "Total vectors needed for test: " << total_vectors_needed << std::endl;
    
    // Open output files
    std::ofstream csv_file(csv_filename);
    std::ofstream log_file(log_filename);
    
    if (!csv_file.is_open() || !log_file.is_open()) {
        std::cerr << "Error: Could not open output files!" << std::endl;
        return 1;
    }
    
    // Write headers
    TestMetrics dummy;
    dummy.writeCSVHeader(csv_file);
    
    log_file << "HNSW Degradation Test Log" << std::endl;
    log_file << "=========================" << std::endl;
    log_file << "Initial vectors: " << initial_vectors << std::endl;
    log_file << "Deletion batch: " << deletion_batch_size << std::endl;
    log_file << "Insertion batch: " << insertion_batch_size << std::endl;
    log_file << "Total vectors needed: " << total_vectors_needed << std::endl;
    log_file << "Iterations: " << num_iterations << std::endl;
    log_file << "HNSW M: " << M << std::endl;
    log_file << "HNSW ef_construction: " << ef_construction << std::endl;
    log_file << "HNSW ef_search: " << std::max(400, initial_vectors / 100) << std::endl;
    log_file << "LSH repair enabled: " << (enable_lsh_repair ? "Yes" : "No") << std::endl;
    if (enable_lsh_repair) {
        log_file << "LSH tables: " << lsh_num_tables << std::endl;
        log_file << "LSH hashes per table: " << lsh_num_hashes << std::endl;
        log_file << "Total LSH hyperplanes: " << (lsh_num_tables * lsh_num_hashes) << std::endl;
    }
    auto now = std::time(nullptr);
    log_file << "Timestamp: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << std::endl;
    log_file << "Fresh vectors strategy: Using sequential dataset vectors (no cycling)" << std::endl;
    log_file << std::endl;
    
    std::cout << "Loading real " << dataset_type << " dataset..." << std::endl;
    
    // RNG for shuffling and fallback synthetic data generation
    std::mt19937 rng(42);
    
    // Load real dataset
    FvecsLoader::Dataset dataset;
    try {
        dataset = FvecsLoader::load_fvecs(dataset_path, total_vectors_needed);
        
        // Verify dimensions match
        if (dataset.dimension != dimension) {
            std::cerr << "Error: Dataset dimension (" << dataset.dimension 
                      << ") doesn't match expected (" << dimension << ")" << std::endl;
            return 1;
        }
        
        // Adjust initial_vectors if dataset is smaller than needed for initial setup
        if (dataset.num_vectors < initial_vectors) {
            std::cout << "Warning: Dataset only has " << dataset.num_vectors 
                      << " vectors, adjusting initial_vectors" << std::endl;
            initial_vectors = dataset.num_vectors;
        }
        
        // Check if we have enough vectors for the entire test
        if (dataset.num_vectors < total_vectors_needed) {
            std::cout << "Warning: Dataset only has " << dataset.num_vectors 
                      << " vectors, but test needs " << total_vectors_needed << std::endl;
            std::cout << "Test will use all available vectors without cycling" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        std::cerr << "Falling back to synthetic data..." << std::endl;
        
        // Fallback to synthetic data - generate enough for entire test
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        dataset.data.resize(total_vectors_needed * dimension);
        dataset.num_vectors = total_vectors_needed;
        dataset.dimension = dimension;
        
        for (int i = 0; i < total_vectors_needed * dimension; i++) {
            dataset.data[i] = dist(rng);
        }
        std::cout << "Generated synthetic " << dataset_type << "-like data" << std::endl;
    }
    
    // Extract base_data for compatibility with existing code
    std::vector<float> base_data = dataset.data;
    
    // Load query vectors from dataset
    std::vector<float> query_data;
    bool using_real_queries = false;
    try {
        std::string query_path = dataset_type == "SIFT" ? 
            "/home/josh/hnswlib/datasets/sift/sift_query.fvecs" : 
            "/home/josh/hnswlib/datasets/gist/gist_query.fvecs";
        
        FvecsLoader::Dataset query_dataset = FvecsLoader::load_fvecs(query_path, 100); // Load 100 queries
        
        if (query_dataset.dimension == dimension) {
            query_data = query_dataset.data;
            using_real_queries = true;
            std::cout << "Loaded " << query_dataset.num_vectors << " real query vectors from dataset" << std::endl;
        } else {
            std::cout << "Warning: Query dimension mismatch, using synthetic queries" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Could not load real queries: " << e.what() << ", using synthetic queries" << std::endl;
    }
    
    // Fallback to synthetic queries if real ones not available
    if (!using_real_queries) {
        std::mt19937 query_rng(1337);  // Fixed seed for identical queries across runs
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        query_data.resize(100 * dimension);
        for (int i = 0; i < 100 * dimension; i++) {
            query_data[i] = dist(query_rng);
        }
        std::cout << "Generated 100 synthetic query vectors" << std::endl;
    }
    
    std::cout << "Building HNSW index..." << std::endl;
    std::cout << "LSH repair enabled: " << (enable_lsh_repair ? "Yes" : "No") << std::endl;
    StopW build_timer;
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(
        &space, initial_vectors * 5, M, ef_construction, 42, enable_replacement, enable_lsh_repair, lsh_num_tables, lsh_num_hashes, lsh_repair_threshold); // 5x capacity for growth

    // Create brute force index for ground truth
    hnswlib::BruteforceSearch<float>* brute_force = new hnswlib::BruteforceSearch<float>(
        &space, initial_vectors * 5); // Match HNSW capacity

    for (int i = 0; i < initial_vectors; i++) {
        index->addPoint(base_data.data() + i * dimension, i);
        brute_force->addPoint(base_data.data() + i * dimension, i);
        if (i % 10000 == 0 && i > 0) {
            std::cout << "  Built " << i << " vectors..." << std::endl;
        }
    }
    double build_time = build_timer.getElapsedTimeMicro() / 1000000.0;
    std::cout << "Index built in " << std::fixed << std::setprecision(2) << build_time << " seconds" << std::endl;
    
    // Set search ef parameter for better recall (scale with dataset size)
    int search_ef = std::max(400, initial_vectors / 100);  // Higher ef for larger datasets
    index->setEf(search_ef);
    std::cout << "Set search ef to: " << search_ef << std::endl;    log_file << "Index build time: " << build_time << " seconds" << std::endl;
    log_file << std::endl;
    
    // Track active labels
    std::vector<hnswlib::labeltype> active_labels;
    for (int i = 0; i < initial_vectors; i++) {
        active_labels.push_back(i);
    }
    hnswlib::labeltype next_label = initial_vectors;
    
    std::cout << "Starting degradation test..." << std::endl;
    TestMetrics::printConsoleHeader();
    
    StopW total_timer;
    StopW iteration_timer;
    
    // Initial measurement
    auto initial_metrics = analyzeGraph(index, brute_force, base_data, query_data, dimension, k, iteration_timer, total_timer, 0);
    initial_metrics.unreachable_deleted_nodes = 0; // No nodes deleted yet
    initial_metrics.printConsole();
    initial_metrics.writeCSVRow(csv_file);
    csv_file.flush();
    
    // Run degradation test
    for (int iter = 1; iter <= num_iterations; iter++) {
        StopW ops_timer; // Timer for just insertion/deletion operations
        
        // Delete oldest elements - FIFO deletion strategy (no shuffling)
        int actual_deletions = std::min(deletion_batch_size, static_cast<int>(active_labels.size()));
        
        int unreachable_deleted_this_iteration = 0;
        std::vector<hnswlib::labeltype> labels_to_delete;
        
        // Collect the oldest labels (from front of vector)
        for (int i = 0; i < actual_deletions; i++) {
            labels_to_delete.push_back(active_labels[i]);
        }
        
        // Process deletions
        for (hnswlib::labeltype label_to_delete : labels_to_delete) {
            
            // Check if this node is unreachable from entry point before deleting it
            hnswlib::tableint internal_id;
            {
                std::unique_lock<std::mutex> lock_table(index->label_lookup_lock);
                auto search = index->label_lookup_.find(label_to_delete);
                if (search == index->label_lookup_.end()) {
                    // Label not found, skip
                    index->markDelete(label_to_delete);
                    brute_force->removePoint(label_to_delete);
                    continue;
                }
                internal_id = search->second;
            }
            
            // Use BFS to check if node is reachable from entry point
            std::unordered_set<hnswlib::tableint> reachable_nodes = findReachableNodes(index);
            
            // If node was unreachable before deletion, count it
            if (reachable_nodes.find(internal_id) == reachable_nodes.end()) {
                unreachable_deleted_this_iteration++;
            }
            
            // Now delete the node
            index->markDelete(label_to_delete);
            brute_force->removePoint(label_to_delete);
        }
        
        // Remove the deleted labels from active_labels
        active_labels.erase(active_labels.begin(), active_labels.begin() + actual_deletions);
        
        // Insert new elements using fresh vectors from the dataset
        for (int i = 0; i < insertion_batch_size; i++) {
            int source_idx = next_label; // Use sequential vectors from dataset
            
            // Check if we still have fresh vectors available
            if (source_idx < dataset.num_vectors) {
                index->addPoint(base_data.data() + source_idx * dimension, next_label, enable_replacement); // Use replacement parameter
                brute_force->addPoint(base_data.data() + source_idx * dimension, next_label);
                active_labels.push_back(next_label);
                next_label++;
            } else {
                // Stop the test if we've exhausted the dataset
                std::cout << "Dataset exhausted at iteration " << iter 
                          << " after inserting " << i << " vectors out of " << insertion_batch_size << std::endl;
                std::cout << "Terminating test early - no more fresh vectors available" << std::endl;
                log_file << "Test terminated early: Dataset exhausted at iteration " << iter << std::endl;
                goto test_complete; // Break out of both loops
            }
        }
        
        double ops_time_seconds = ops_timer.getElapsedTimeMicro() / 1000000.0; // Capture ops time
        
        // Analyze and log (metrics computation not included in ops timing)
        iteration_timer.reset(); // Reset for compatibility with analyzeGraph
        auto metrics = analyzeGraph(index, brute_force, base_data, query_data, dimension, k, iteration_timer, total_timer, iter);
        metrics.iteration_time_seconds = ops_time_seconds; // Override with ops-only timing
        metrics.unreachable_deleted_nodes = unreachable_deleted_this_iteration; // Set per-iteration unreachable count
        metrics.printConsole();
        metrics.writeCSVRow(csv_file);
        csv_file.flush();
        
        // Log significant events
        if (metrics.disconnected_nodes > 0) {
            log_file << "Iteration " << iter << ": " << metrics.disconnected_nodes << " disconnected nodes detected" << std::endl;
        }
        
        if (metrics.recall < 0.5 && iter > 5) {
            log_file << "Iteration " << iter << ": Low recall detected: " << metrics.recall << std::endl;
        }
        
        // Progress reporting
        if (iter % 10 == 0) {
            std::cout << "  [Progress: " << iter << "/" << num_iterations 
                      << ", Elapsed: " << std::fixed << std::setprecision(1) 
                      << metrics.cumulative_time_seconds << "s]" << std::endl;
            log_file << "Progress checkpoint at iteration " << iter << std::endl;
        }
        
        // Early termination if severely degraded
        if (metrics.recall < 0.1 && iter > 10) {
            std::cout << "\nEarly termination: Severe degradation detected" << std::endl;
            log_file << "Early termination at iteration " << iter << ": severe degradation" << std::endl;
            break;
        }
    }
    
test_complete:
    double total_time = total_timer.getElapsedTimeMicro() / 1000000.0;
    std::cout << "\nTest completed in " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
    std::cout << "Results saved to:" << std::endl;
    std::cout << "  CSV data: " << csv_filename << std::endl;
    std::cout << "  Log file: " << log_filename << std::endl;
    
    log_file << std::endl;
    log_file << "Test completed in " << total_time << " seconds" << std::endl;
    log_file << "Final active elements: " << active_labels.size() << std::endl;
    log_file << "Final deleted elements: " << index->getDeletedCount() << std::endl;
    
    // Close files
    csv_file.close();
    log_file.close();
    
    delete index;
    delete brute_force;
    return 0;
}
