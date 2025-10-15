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

// Simple timing test that measures only iteration time and final recall
// Usage: ./degradation_test_timing_only dimension initial_vectors deletion_batch insertion_batch num_iterations k M ef_construction lsh_threshold enable_lsh enable_replacement

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

struct SimpleTestMetrics {
    int iteration;
    double iteration_time_seconds;
    double cumulative_time_seconds;
    
    void writeCSVHeader(std::ofstream& file) {
        file << "iteration,iteration_time_seconds,cumulative_time_seconds\n";
    }
    
    void writeCSVRow(std::ofstream& file) {
        file << iteration << ","
             << std::fixed << std::setprecision(6) << iteration_time_seconds << ","
             << std::fixed << std::setprecision(6) << cumulative_time_seconds << "\n";
    }
    
    void printConsole() {
        std::cout << std::setw(4) << iteration 
                  << std::setw(15) << std::fixed << std::setprecision(6) << iteration_time_seconds
                  << std::setw(15) << std::fixed << std::setprecision(6) << cumulative_time_seconds
                  << std::endl;
    }
    
    static void printConsoleHeader() {
        std::cout << std::setw(4) << "Iter"
                  << std::setw(15) << "IterTime(s)"
                  << std::setw(15) << "TotalTime(s)"
                  << std::endl;
    }
};

struct FinalRecallMetrics {
    double final_recall;
    double final_search_time_ms;
    int final_active_elements;
    int final_deleted_elements;
    
    void writeToFile(const std::string& filename) {
        std::ofstream file(filename);
        file << "metric,value\n";
        file << "final_recall," << std::fixed << std::setprecision(6) << final_recall << "\n";
        file << "final_search_time_ms," << std::fixed << std::setprecision(2) << final_search_time_ms << "\n";
        file << "final_active_elements," << final_active_elements << "\n";
        file << "final_deleted_elements," << final_deleted_elements << "\n";
    }
    
    void printConsole() {
        std::cout << "\nFinal Results:" << std::endl;
        std::cout << "  Recall: " << std::fixed << std::setprecision(4) << final_recall << std::endl;
        std::cout << "  Search time: " << std::fixed << std::setprecision(2) << final_search_time_ms << " ms" << std::endl;
        std::cout << "  Active elements: " << final_active_elements << std::endl;
        std::cout << "  Deleted elements: " << final_deleted_elements << std::endl;
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

// Calculate recall only after all operations are complete
FinalRecallMetrics calculateFinalRecall(hnswlib::HierarchicalNSW<float>* index, 
                                        hnswlib::BruteforceSearch<float>* brute_force,
                                        const std::vector<float>& query_data,
                                        int dimension, int k) {
    FinalRecallMetrics metrics;
    
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
    
    metrics.final_recall = total_results > 0 ? static_cast<double>(total_correct) / total_results : 0.0;
    metrics.final_search_time_ms = total_hnsw_search_time_micro / 1000.0;
    metrics.final_active_elements = index->cur_element_count - index->getDeletedCount();
    metrics.final_deleted_elements = index->getDeletedCount();
    
    return metrics;
}

int main(int argc, char* argv[]) {
    std::cout << "HNSW Degradation Test - Timing Only" << std::endl;
    std::cout << "====================================" << std::endl;
    
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
    int lsh_num_tables = 6;
    int lsh_num_hashes = 12;
    
    // Generate output filenames based on LSH configuration and dataset
    std::string base_filename;
    if (enable_lsh_repair) {
        base_filename = "timing_test_" + dataset_type + "_lsh_" + std::to_string(lsh_num_tables) + "x" + std::to_string(lsh_num_hashes);
    } else {
        base_filename = "timing_test_" + dataset_type + "_vanilla_hnsw";
    }
    if (enable_replacement) {
        base_filename += "_with_replacement";
    } else {
        base_filename += "_without_replacement";
    }
    
    std::string timing_csv_filename = generateFilename(base_filename + "_timing", "csv");
    std::string recall_csv_filename = generateFilename(base_filename + "_final_recall", "csv");
    
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
    std::cout << "  Timing output: " << timing_csv_filename << std::endl;
    std::cout << "  Final recall output: " << recall_csv_filename << std::endl;
    std::cout << std::endl;
    
    // Calculate total vectors needed: initial + (iterations * insertion_batch_size)
    int total_vectors_needed = initial_vectors + (num_iterations * insertion_batch_size);
    std::cout << "Total vectors needed for test: " << total_vectors_needed << std::endl;
    
    // Open timing output file
    std::ofstream timing_csv_file(timing_csv_filename);
    if (!timing_csv_file.is_open()) {
        std::cerr << "Error: Could not open timing output file!" << std::endl;
        return 1;
    }
    
    // Write timing CSV header
    SimpleTestMetrics dummy;
    dummy.writeCSVHeader(timing_csv_file);
    
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

    // Create brute force index for ground truth (only used for final recall calculation)
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
    std::cout << "Set search ef to: " << search_ef << std::endl;
    
    // Track active labels
    std::vector<hnswlib::labeltype> active_labels;
    for (int i = 0; i < initial_vectors; i++) {
        active_labels.push_back(i);
    }
    hnswlib::labeltype next_label = initial_vectors;
    
    std::cout << "Starting timing test..." << std::endl;
    SimpleTestMetrics::printConsoleHeader();
    
    StopW total_timer;
    
    // Initial timing entry (iteration 0 - no operations, just setup time)
    SimpleTestMetrics initial_metrics;
    initial_metrics.iteration = 0;
    initial_metrics.iteration_time_seconds = 0.0;
    initial_metrics.cumulative_time_seconds = 0.0;
    initial_metrics.printConsole();
    initial_metrics.writeCSVRow(timing_csv_file);
    timing_csv_file.flush();
    
    // Run degradation test - measure only operation timing
    for (int iter = 1; iter <= num_iterations; iter++) {
        StopW iteration_timer; // Timer for this iteration's operations
        
        // Delete oldest elements - FIFO deletion strategy (no shuffling)
        int actual_deletions = std::min(deletion_batch_size, static_cast<int>(active_labels.size()));
        
        std::vector<hnswlib::labeltype> labels_to_delete;
        
        // Collect the oldest labels (from front of vector)
        for (int i = 0; i < actual_deletions; i++) {
            labels_to_delete.push_back(active_labels[i]);
        }
        
        // Process deletions
        for (hnswlib::labeltype label_to_delete : labels_to_delete) {
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
                goto test_complete; // Break out of both loops
            }
        }
        
        // Record timing metrics
        SimpleTestMetrics metrics;
        metrics.iteration = iter;
        metrics.iteration_time_seconds = iteration_timer.getElapsedTimeMicro() / 1000000.0;
        metrics.cumulative_time_seconds = total_timer.getElapsedTimeMicro() / 1000000.0;
        metrics.printConsole();
        metrics.writeCSVRow(timing_csv_file);
        timing_csv_file.flush();
        
        // Progress reporting
        if (iter % 10 == 0) {
            std::cout << "  [Progress: " << iter << "/" << num_iterations 
                      << ", Elapsed: " << std::fixed << std::setprecision(1) 
                      << metrics.cumulative_time_seconds << "s]" << std::endl;
        }
    }
    
test_complete:
    double total_time = total_timer.getElapsedTimeMicro() / 1000000.0;
    timing_csv_file.close();
    
    std::cout << "\nTiming test completed in " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
    std::cout << "Now calculating final recall..." << std::endl;
    
    // Calculate final recall after all operations are complete
    auto final_recall_metrics = calculateFinalRecall(index, brute_force, query_data, dimension, k);
    final_recall_metrics.writeToFile(recall_csv_filename);
    final_recall_metrics.printConsole();
    
    std::cout << "\nResults saved to:" << std::endl;
    std::cout << "  Timing data: " << timing_csv_filename << std::endl;
    std::cout << "  Final recall: " << recall_csv_filename << std::endl;
    
    delete index;
    delete brute_force;
    return 0;
}