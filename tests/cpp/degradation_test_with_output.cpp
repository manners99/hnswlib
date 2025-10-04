#include "../../hnswlib/hnswlib.h"
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
    
    void writeCSVHeader(std::ofstream& file) {
        file << "iteration,active_elements,deleted_elements,recall,search_time_ms,"
             << "total_connections,disconnected_nodes,avg_inbound,min_inbound,max_inbound,"
             << "connectivity_density,iteration_time_seconds,cumulative_time_seconds\n";
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
             << std::setprecision(2) << cumulative_time_seconds << "\n";
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
    StopW search_timer;
    int total_queries = 100;
    int total_correct = 0;
    int total_results = 0;
    
    for (int i = 0; i < total_queries; i++) {
        // Get ground truth from brute force
        auto gt_result = brute_force->searchKnn(query_data.data() + i * dimension, k);
        std::unordered_set<hnswlib::labeltype> gt_set;
        while (!gt_result.empty()) {
            gt_set.insert(gt_result.top().second);
            gt_result.pop();
        }
        
        // Get HNSW results
        auto hnsw_result = index->searchKnn(query_data.data() + i * dimension, k);
        
        // Count correct results
        while (!hnsw_result.empty()) {
            if (gt_set.find(hnsw_result.top().second) != gt_set.end()) {
                total_correct++;
            }
            hnsw_result.pop();
            total_results++;
        }
    }
    
    metrics.recall = total_results > 0 ? static_cast<double>(total_correct) / total_results : 0.0;
    metrics.search_time_ms = search_timer.getElapsedTimeMicro() / 1000.0;
    
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
    
    return metrics;
}

int main(int argc, char* argv[]) {
    std::cout << "HNSW Degradation Test with Data Logging" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Test parameters - configurable via command line
    int initial_vectors = 100000;     // Default 100K
    int deletion_batch_size = 10000;  // Default 10K
    int insertion_batch_size = 10000; // Default 10K
    int num_iterations = 100;         // Default 100
    int dimension = 128;              // SIFT dimension
    int k = 10;
    
    // Parse command line arguments
    if (argc >= 2) initial_vectors = std::atoi(argv[1]);
    if (argc >= 3) deletion_batch_size = std::atoi(argv[2]);
    if (argc >= 4) insertion_batch_size = std::atoi(argv[3]);
    if (argc >= 5) num_iterations = std::atoi(argv[4]);
    
    // HNSW parameters
    int M = 16;
    int ef_construction = 200;
    
    // Generate output filenames
    std::string csv_filename = generateFilename("degradation_test", "csv");
    std::string log_filename = generateFilename("degradation_test", "log");
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Initial vectors: " << initial_vectors << std::endl;
    std::cout << "  Deletion batch: " << deletion_batch_size << std::endl;
    std::cout << "  Insertion batch: " << insertion_batch_size << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW ef_construction: " << ef_construction << std::endl;
    std::cout << "  HNSW ef_search: " << std::max(400, initial_vectors / 100) << std::endl;
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
    auto now = std::time(nullptr);
    log_file << "Timestamp: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << std::endl;
    log_file << std::endl;
    
    std::cout << "Generating synthetic SIFT-like data..." << std::endl;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Match Python: [0,1] range
    
    // Generate base data
    std::vector<float> base_data(initial_vectors * dimension);
    for (int i = 0; i < initial_vectors * dimension; i++) {
        base_data[i] = dist(rng);
    }
    
    // Query data is now generated fresh each iteration in analyzeGraph()
    
    std::cout << "Building HNSW index..." << std::endl;
    bool enable_lsh_repair = true;
    std::cout << enable_lsh_repair << std::endl;
    StopW build_timer;
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(
        &space, initial_vectors * 2, M, ef_construction, 42, true, enable_lsh_repair, 10, 10);
        //                                                         enable_lsh_repair = true, tables, hashes

    // Create brute force index for ground truth
    hnswlib::BruteforceSearch<float>* brute_force = new hnswlib::BruteforceSearch<float>(
        &space, initial_vectors * 2);

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
    auto initial_metrics = analyzeGraph(index, brute_force, base_data, dimension, k, iteration_timer, total_timer, 0);
    initial_metrics.printConsole();
    initial_metrics.writeCSVRow(csv_file);
    csv_file.flush();
    
    // Run degradation test
    for (int iter = 1; iter <= num_iterations; iter++) {
        iteration_timer.reset();
        
        // Delete elements
        std::shuffle(active_labels.begin(), active_labels.end(), rng);
        int actual_deletions = std::min(deletion_batch_size, static_cast<int>(active_labels.size()));
        
        for (int i = 0; i < actual_deletions; i++) {
            index->markDelete(active_labels.back());
            brute_force->removePoint(active_labels.back());
            active_labels.pop_back();
        }
        
        // Insert new elements (reuse base data cyclically)
        for (int i = 0; i < insertion_batch_size; i++) {
            int source_idx = (next_label - initial_vectors) % initial_vectors;
            index->addPoint(base_data.data() + source_idx * dimension, next_label, true);
            brute_force->addPoint(base_data.data() + source_idx * dimension, next_label);
            active_labels.push_back(next_label);
            next_label++;
        }
        
        // Analyze and log
        auto metrics = analyzeGraph(index, brute_force, base_data, dimension, k, iteration_timer, total_timer, iter);
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
