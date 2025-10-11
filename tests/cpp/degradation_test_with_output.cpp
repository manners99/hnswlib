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
    
    // Deletion statistics
    int unreachable_deletions;
    
    void writeCSVHeader(std::ofstream& file) {
        file << "iteration,active_elements,deleted_elements,recall,search_time_ms,"
             << "total_connections,disconnected_nodes,avg_inbound,min_inbound,max_inbound,"
             << "connectivity_density,iteration_time_seconds,cumulative_time_seconds,"
             << "lsh_repair_calls,lsh_repairs_performed,lsh_queries_made,lsh_connections_added,"
             << "unreachable_deletions\n";
    }
    
    void writeCSVRow(std::ofstream& file) {
        file << iteration << "," << active_elements << "," << deleted_elements << ","
             << std::fixed << std::setprecision(4) << recall << ","
             << std::setprecision(2) << search_time_ms << ","
             << total_connections << "," << disconnected_nodes << ","
             << std::setprecision(2) << avg_inbound_connections << ","
             << min_inbound_connections << "," << max_inbound_connections << ","
             << std::setprecision(4) << connectivity_density << ","
             << std::setprecision(2) << iteration_time_seconds << ","
             << std::setprecision(2) << cumulative_time_seconds << ","
             << lsh_repair_calls << "," << lsh_repairs_performed << ","
             << lsh_queries_made << "," << lsh_connections_added << ","
             << unreachable_deletions << "\n";
    }
    
    void printConsole() {
        std::cout << std::setw(4) << iteration 
                  << std::setw(8) << active_elements 
                  << std::setw(8) << deleted_elements
                  << std::setw(8) << std::fixed << std::setprecision(3) << recall
                  << std::setw(10) << std::setprecision(1) << search_time_ms
                  << std::setw(8) << total_connections
                  << std::setw(6) << disconnected_nodes
                  << std::setw(8) << std::setprecision(1) << avg_inbound_connections
                  << std::setw(6) << lsh_repairs_performed
                  << std::setw(6) << lsh_connections_added
                  << std::setw(6) << unreachable_deletions
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
                  << std::setw(8) << "AvgIn"
                  << std::setw(6) << "Repair"
                  << std::setw(6) << "NewCon"
                  << std::setw(6) << "UnrDel"
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

// Count unreachable (disconnected) points in the graph
int countUnreachablePoints(hnswlib::HierarchicalNSW<float>* index) {
    int cur_element_count = index->cur_element_count;
    std::vector<int> inbound_connections(cur_element_count, 0);
    
    // Count inbound connections for all active nodes
    for (int i = 0; i < cur_element_count; i++) {
        if (index->isMarkedDeleted(i)) continue;
        
        for (int l = 0; l <= index->element_levels_[i]; l++) {
            auto ll_cur = index->get_linklist_at_level(i, l);
            int size = index->getListCount(ll_cur);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
            
            for (int j = 0; j < size; j++) {
                if (data[j] < cur_element_count && !index->isMarkedDeleted(data[j])) {
                    inbound_connections[data[j]]++;
                }
            }
        }
    }
    
    // Count nodes with zero inbound connections (unreachable)
    int unreachable_count = 0;
    for (int i = 0; i < cur_element_count; i++) {
        if (!index->isMarkedDeleted(i) && inbound_connections[i] == 0) {
            unreachable_count++;
        }
    }
    
    return unreachable_count;
}

// Get set of all unreachable points (points with no inbound connections)
std::unordered_set<hnswlib::labeltype> getUnreachablePoints(hnswlib::HierarchicalNSW<float>* index) {
    int cur_element_count = index->cur_element_count;
    std::vector<int> inbound_connections(cur_element_count, 0);
    
    // Count inbound connections for all active nodes
    for (int i = 0; i < cur_element_count; i++) {
        if (index->isMarkedDeleted(i)) continue;
        
        for (int l = 0; l <= index->element_levels_[i]; l++) {
            auto ll_cur = index->get_linklist_at_level(i, l);
            int size = index->getListCount(ll_cur);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
            
            for (int j = 0; j < size; j++) {
                if (data[j] < cur_element_count && !index->isMarkedDeleted(data[j])) {
                    inbound_connections[data[j]]++;
                }
            }
        }
    }
    
    // Collect nodes with zero inbound connections (unreachable)
    std::unordered_set<hnswlib::labeltype> unreachable_points;
    for (int i = 0; i < cur_element_count; i++) {
        if (!index->isMarkedDeleted(i) && inbound_connections[i] == 0) {
            unreachable_points.insert(i);
        }
    }
    
    return unreachable_points;
}

TestMetrics analyzeGraph(hnswlib::HierarchicalNSW<float>* index, 
                        hnswlib::BruteforceSearch<float>* brute_force,
                        const std::vector<float>& base_data,
                        int dimension, int k, 
                        StopW& iteration_timer, StopW& total_timer,
                        int iteration) {
    TestMetrics metrics;
    metrics.iteration = iteration;
    
    // Generate query data with fixed seed for consistent comparison
    std::mt19937 rng(1337);  // Fixed seed for identical queries across runs
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> query_data(100 * dimension);
    for (int i = 0; i < 100 * dimension; i++) {
        query_data[i] = dist(rng);  // Fixed queries for comparison consistency
    }
    
    // Measure search performance with proper ground truth from brute force
    int total_queries = 100;
    int total_correct = 0;
    int total_results = 0;
    double total_hnsw_search_time_micro = 0.0;
    
    // Adjust k if we have fewer elements than k
    int active_count = index->cur_element_count - index->getDeletedCount();
    int effective_k = std::min(k, active_count);
    
    if (effective_k <= 0) {
        metrics.recall = 0.0;
        metrics.search_time_ms = 0.0;
        return metrics; // Skip if no active elements
    }
    
    for (int i = 0; i < total_queries; i++) {
        // Get ground truth from brute force (not timed)
        auto gt_result = brute_force->searchKnn(query_data.data() + i * dimension, effective_k);
        std::unordered_set<hnswlib::labeltype> gt_set;
        while (!gt_result.empty()) {
            gt_set.insert(gt_result.top().second);
            gt_result.pop();
        }
        
        // Time only the HNSW search
        StopW hnsw_search_timer;
        auto hnsw_result = index->searchKnn(query_data.data() + i * dimension, effective_k);
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
    
    std::vector<int> inbound_connections(cur_element_count, 0);
    int total_connections = 0;
    int disconnected = 0;
    
    // Count all connections
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
    
    // Calculate connection statistics
    std::vector<int> valid_inbound;
    for (int i = 0; i < cur_element_count; i++) {
        if (!index->isMarkedDeleted(i)) {
            valid_inbound.push_back(inbound_connections[i]);
            if (inbound_connections[i] == 0) disconnected++;
        }
    }
    
    metrics.total_connections = total_connections;
    metrics.disconnected_nodes = disconnected;
    
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
    bool allow_replace_deleted = false; // Default no replace deleted
    
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
    if (argc >= 12) allow_replace_deleted = (std::atoi(argv[11]) != 0);
    
    // Determine dataset type based on dimension
    std::string dataset_type = (dimension == 128) ? "SIFT" : "GIST";
    std::string dataset_path;
    
    if (dataset_type == "SIFT") {
        dataset_path = "../../datasets/sift/sift_base.fvecs";
    } else {
        dataset_path = "../../datasets/gist/gist_base.fvecs";
    }
    
    // LSH configuration (fixed for now, could be parameterized later)
    int lsh_num_tables = 8;
    int lsh_num_hashes = 10;
    
    // Generate output filenames based on configuration
    std::string base_filename;
    if (enable_lsh_repair) {
        base_filename = "degradation_test_" + dataset_type + "_lsh_" + std::to_string(lsh_num_tables) + "x" + std::to_string(lsh_num_hashes) + "_thresh" + std::to_string(lsh_repair_threshold);
    } else {
        base_filename = "degradation_test_" + dataset_type + "_vanilla_hnsw";
    }
    if (allow_replace_deleted) {
        base_filename += "_replace";
    } else {
        base_filename += "_noreplace";
    }
    std::string csv_filename = generateFilename(base_filename, "csv");
    std::string log_filename = generateFilename(base_filename, "log");
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dataset: " << dataset_type << " (real data)" << std::endl;
    std::cout << "  Dimension: " << dimension << std::endl;
    std::cout << "  Initial vectors: " << initial_vectors << std::endl;
    std::cout << "  Deletion batch: " << deletion_batch_size << std::endl;
    std::cout << "  Insertion batch: " << insertion_batch_size << " (using fresh data)" << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  k (recall test): " << k << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW ef_construction: " << ef_construction << std::endl;
    std::cout << "  HNSW ef_search: " << std::max(400, initial_vectors / 100) << std::endl;
    std::cout << "  LSH repair enabled: " << (enable_lsh_repair ? "Yes" : "No") << std::endl;
    if (enable_lsh_repair) {
        std::cout << "  LSH tables: " << lsh_num_tables << std::endl;
        std::cout << "  LSH hashes per table: " << lsh_num_hashes << std::endl;
        std::cout << "  LSH repair threshold: " << lsh_repair_threshold << std::endl;
    }
    std::cout << "  Replace deleted enabled: " << (allow_replace_deleted ? "Yes" : "No") << std::endl;
    std::cout << "  Output CSV: " << csv_filename << std::endl;
    std::cout << "  Output log: " << log_filename << std::endl;
    std::cout << std::endl;
    
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
    log_file << std::endl;
    
    std::cout << "Loading real " << dataset_type << " dataset..." << std::endl;
    
    // RNG for shuffling and fallback synthetic data generation
    std::mt19937 rng(42);
    
    // Load real dataset
    FvecsLoader::Dataset dataset;
    try {
        // Load more vectors than initial to have fresh data for insertions
        int estimated_needed = initial_vectors + (insertion_batch_size * num_iterations);
        int vectors_to_load = std::min(estimated_needed, 1000000); // Cap at 1M for memory
        
        dataset = FvecsLoader::load_fvecs(dataset_path, vectors_to_load);
        
        // Verify dimensions match
        if (dataset.dimension != dimension) {
            std::cerr << "Error: Dataset dimension (" << dataset.dimension 
                      << ") doesn't match expected (" << dimension << ")" << std::endl;
            return 1;
        }
        
        // Adjust initial_vectors if dataset is smaller
        if (dataset.num_vectors < initial_vectors) {
            std::cout << "Warning: Dataset only has " << dataset.num_vectors 
                      << " vectors, adjusting initial_vectors" << std::endl;
            initial_vectors = dataset.num_vectors;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        std::cerr << "Falling back to synthetic data..." << std::endl;
        
        // Fallback to synthetic data
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        dataset.data.resize(initial_vectors * dimension);
        dataset.num_vectors = initial_vectors;
        dataset.dimension = dimension;
        
        for (int i = 0; i < initial_vectors * dimension; i++) {
            dataset.data[i] = dist(rng);
        }
        std::cout << "Generated synthetic " << dataset_type << "-like data" << std::endl;
    }
    
    // Extract base_data for compatibility with existing code
    std::vector<float> base_data = dataset.data;
    
    // Calculate fresh data availability
    int total_vectors_available = base_data.size() / dimension;
    int fresh_vectors_available = total_vectors_available - initial_vectors;
    int max_fresh_iterations = (insertion_batch_size > 0) ? fresh_vectors_available / insertion_batch_size : 0;
    
    std::cout << "Fresh data availability:" << std::endl;
    std::cout << "  Total vectors in dataset: " << total_vectors_available << std::endl;
    std::cout << "  Fresh vectors available for insertion: " << fresh_vectors_available << std::endl;
    std::cout << "  Iterations with fresh data: " << max_fresh_iterations << " / " << num_iterations << std::endl;
    std::cout << std::endl;
    
    // Query data is now generated fresh each iteration in analyzeGraph()
    
    std::cout << "Building HNSW index..." << std::endl;
    std::cout << "LSH repair enabled: " << (enable_lsh_repair ? "Yes" : "No") << std::endl;
    StopW build_timer;
    hnswlib::L2Space space(dimension);
    
    // Calculate appropriate capacity based on replacement mode
    int capacity_multiplier = allow_replace_deleted ? 2 : 
                             std::max(5, 1 + (num_iterations * insertion_batch_size) / initial_vectors);
    int index_capacity = initial_vectors * capacity_multiplier;
    std::cout << "Index capacity: " << index_capacity << " (multiplier: " << capacity_multiplier << ")" << std::endl;
    
    hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(
        &space, index_capacity, M, ef_construction, 42, allow_replace_deleted, enable_lsh_repair, lsh_num_tables, lsh_num_hashes, lsh_repair_threshold);

    // Create brute force index for ground truth
    hnswlib::BruteforceSearch<float>* brute_force = new hnswlib::BruteforceSearch<float>(
        &space, index_capacity);

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
    int next_data_index = initial_vectors;  // Track position in dataset for fresh insertions
    
    // Track unreachable points from previous iteration
    std::unordered_set<hnswlib::labeltype> previous_unreachable_points;
    
    std::cout << "Starting degradation test..." << std::endl;
    TestMetrics::printConsoleHeader();
    
    StopW total_timer;
    StopW iteration_timer;
    
    // Initial measurement
    auto initial_metrics = analyzeGraph(index, brute_force, base_data, dimension, k, iteration_timer, total_timer, 0);
    initial_metrics.unreachable_deletions = 0; // No deletions in initial measurement
    initial_metrics.printConsole();
    initial_metrics.writeCSVRow(csv_file);
    csv_file.flush();
    
    // Run degradation test
    for (int iter = 1; iter <= num_iterations; iter++) {
        StopW ops_timer; // Timer for just insertion/deletion operations
        
        // Delete elements and track previously unreachable points being deleted
        std::shuffle(active_labels.begin(), active_labels.end(), rng);
        int actual_deletions = std::min(deletion_batch_size, static_cast<int>(active_labels.size()));
        
        int unreachable_deletions = 0; // Count previously unreachable points being deleted
        
        for (int i = 0; i < actual_deletions; i++) {
            hnswlib::labeltype label_to_delete = active_labels.back();
            
            // Check if this point was unreachable in previous iteration
            if (previous_unreachable_points.find(label_to_delete) != previous_unreachable_points.end()) {
                unreachable_deletions++;
            }
            
            index->markDelete(label_to_delete);
            brute_force->removePoint(label_to_delete);
            active_labels.pop_back();
        }        // Insert new elements (use fresh data from dataset)
        int actual_insertions = insertion_batch_size;
        
        // Check if we have enough fresh data left in the dataset
        if (next_data_index + insertion_batch_size > base_data.size() / dimension) {
            actual_insertions = (base_data.size() / dimension) - next_data_index;
            if (actual_insertions <= 0) {
                std::cout << "Warning: No more fresh data available for insertion at iteration " << iter << std::endl;
                actual_insertions = 0;
            }
        }
        
        for (int i = 0; i < actual_insertions; i++) {
            index->addPoint(base_data.data() + next_data_index * dimension, next_label, allow_replace_deleted);
            brute_force->addPoint(base_data.data() + next_data_index * dimension, next_label);
            active_labels.push_back(next_label);
            next_label++;
            next_data_index++;
        }
        
        double ops_time_seconds = ops_timer.getElapsedTimeMicro() / 1000000.0; // Capture ops time
        
        // Analyze and log (metrics computation not included in ops timing)
        iteration_timer.reset(); // Reset for compatibility with analyzeGraph
        auto metrics = analyzeGraph(index, brute_force, base_data, dimension, k, iteration_timer, total_timer, iter);
        metrics.iteration_time_seconds = ops_time_seconds; // Override with ops-only timing
        metrics.unreachable_deletions = unreachable_deletions; // Add unreachable deletions count
        
        // Update unreachable points list for next iteration (after all operations)
        auto current_unreachable = getUnreachablePoints(index);
        if (unreachable_deletions > 0) {
            std::cout << "  Iteration " << iter << ": " << unreachable_deletions 
                      << " previously unreachable points were deleted out of " << actual_deletions << " total deletions" << std::endl;
        }
        
        metrics.printConsole();
        metrics.writeCSVRow(csv_file);
        csv_file.flush();
        
        // Update previous unreachable points for next iteration
        previous_unreachable_points = current_unreachable;
        
        // Log significant events
        if (metrics.disconnected_nodes > 0) {
            log_file << "Iteration " << iter << ": " << metrics.disconnected_nodes << " disconnected nodes detected" << std::endl;
        }
        
        if (unreachable_deletions > 0) {
            log_file << "Iteration " << iter << ": " << unreachable_deletions 
                     << " previously unreachable points were deleted out of " << actual_deletions << " total deletions" << std::endl;
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
