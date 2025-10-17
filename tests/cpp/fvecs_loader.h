#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

class FvecsLoader {
public:
    struct Dataset {
        std::vector<float> data;
        int num_vectors;
        int dimension;
        
        const float* get_vector(int index) const {
            return data.data() + index * dimension;
        }
    };
    
    static Dataset load_fvecs(const std::string& filename, int max_vectors = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        Dataset dataset;
        dataset.num_vectors = 0;
        dataset.dimension = 0;
        
        // First pass: count vectors and get dimension
        int dim;
        while (file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
            if (dataset.dimension == 0) {
                dataset.dimension = dim;
            } else if (dataset.dimension != dim) {
                throw std::runtime_error("Inconsistent dimensions in file: " + filename);
            }
            
            // Skip vector data
            file.seekg(dim * sizeof(float), std::ios::cur);
            dataset.num_vectors++;
            
            if (max_vectors > 0 && dataset.num_vectors >= max_vectors) {
                break;
            }
        }
        
        if (dataset.num_vectors == 0) {
            throw std::runtime_error("No vectors found in file: " + filename);
        }
        
        // Reset file and load actual data
        file.clear();
        file.seekg(0, std::ios::beg);
        
        dataset.data.reserve(dataset.num_vectors * dataset.dimension);
        
        for (int i = 0; i < dataset.num_vectors; i++) {
            int dim;
            if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
                throw std::runtime_error("Error reading dimension for vector " + std::to_string(i));
            }
            
            std::vector<float> vector(dim);
            if (!file.read(reinterpret_cast<char*>(vector.data()), dim * sizeof(float))) {
                throw std::runtime_error("Error reading vector " + std::to_string(i));
            }
            
            dataset.data.insert(dataset.data.end(), vector.begin(), vector.end());
        }
        
        std::cout << "Loaded " << dataset.num_vectors << " vectors of dimension " 
                  << dataset.dimension << " from " << filename << std::endl;
        
        return dataset;
    }
    
    static void save_fvecs(const std::string& filename, const std::vector<float>& data, 
                          int num_vectors, int dimension) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }
        
        for (int i = 0; i < num_vectors; i++) {
            // Write dimension
            file.write(reinterpret_cast<const char*>(&dimension), sizeof(int));
            
            // Write vector data
            const float* vector_data = data.data() + i * dimension;
            file.write(reinterpret_cast<const char*>(vector_data), dimension * sizeof(float));
        }
        
        std::cout << "Saved " << num_vectors << " vectors of dimension " 
                  << dimension << " to " << filename << std::endl;
    }
};
