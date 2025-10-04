#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <tuple>
#include <functional>
#include <set>
#include <eigen3/Eigen/Dense>
#include <cmath>
//use "sudo apt-get install libeigen3-dev" to intall Eigen library

namespace diskann
{
    //Hash key type is vector<int> representing a binary hash
    using HashKey = std::vector<int>;
    struct HashKeyHash {
        std::size_t operator()(const HashKey& key) const {
            std::size_t h = 0;
            for (int bit : key) {
                h = (h << 1) ^ std::hash<int>()(bit);
            }
            return h;
        }
    };

    struct HashKeyEq {
        bool operator()(const HashKey& a, const HashKey& b) const {
            return a == b;
        }
    };

    class LSH {
    private:
        int numTables;
        int numHashes;
        int dimension;

        //For each table create a map from the hash key to a list of ids
        std::vector<std::unordered_map<HashKey, std::vector<int>, HashKeyHash, HashKeyEq>> tables;
        
        //Hyperplanes: One matix per table (numHashes * dimension)
        std::vector<Eigen::MatrixXd> hyperplanes;

        std::mt19937 gen;
        std::normal_distribution<double> dist;

    public:
        LSH(int numTables, int numHashes, int dimension)
            : numTables(numTables), numHashes(numHashes), dimension(dimension), tables(numTables), gen(std::random_device{}()), dist(0.0, 1.0)
            {
                // std::cerr << "LSH constructor called: tables=" << numTables 
                // << ", numHashes=" << numHashes 
                // << ", dimension=" << dimension << std::endl;

                //reserve the outer vectores
                tables.reserve(numTables);
                //Generate the random hyperplanes
                for (int i = 0; i < numTables; i++) {

                    //Allocate hash table and reserve expected number of buckets
                    std::unordered_map<HashKey, std::vector<int>, HashKeyHash, HashKeyEq> table;
                    size_t numBuckets = 1ULL << numHashes;
                    if (numBuckets > 1e6) 
                        numBuckets = 1e6;

                    table.reserve(numBuckets); // reserve on local table
                    tables.push_back(std::move(table)); // push into tables vector
                    // std::cout << "[SPACE] Reserved " << (numBuckets) << " unique hash keys" << std::endl;

                    //Create the hyperplanes
                    Eigen::MatrixXd hp(numHashes, dimension);
                    for (int j = 0; j < numHashes; j++ ) {
                        for (int k = 0; k < dimension; k++) {
                            hp(j, k) = dist(gen);
                        }
                    }
                    hyperplanes.push_back(hp);
                }

                // std::cout << "[INIT] Created LSH with " << numTables
                // << " tables, " << numHashes
                // << " hashes per table, dimension " << dimension << std::endl;
            }

            HashKey hashFunction(const Eigen::VectorXd& point, const Eigen::MatrixXd& hyperplane){
                Eigen::VectorXd projection = hyperplane * point;
                HashKey hash;
                for (int i = 0; i < projection.size(); i++){
                    hash.push_back(projection(i) > 0 ? 1 : 0);
                }
                return hash;
            }

            std::vector<HashKey> generateHash(const Eigen::VectorXd& point) {
                // std::cout << "[GENERATE_HASH] Generating hash for point " << point.transpose() << std::endl;
                std::vector<HashKey> hashes;
                for (int i = 0; i < numTables; i++) {
                    HashKey h = hashFunction(point, hyperplanes[i]);
                    //std::cout << "  [TABLE " << i << "] hash=(";
                    //for (int bit : h) std::cout << bit;
                    //std::cout << ")" << std::endl;
                    hashes.push_back(h);
                }
                return hashes;
            }

            void add(const Eigen::VectorXd& point, int id) {
                //std::cout << "[ADD] Adding point " << id << ": " << point.transpose() << std::endl;
                std::vector<HashKey> hashes = generateHash(point);
                for (int i = 0; i < numTables; i++) {
                    tables[i][hashes[i]].push_back(id);
                    // std::cout << "  [TABLE " << i << "] Inserted " << id << " into bucket (";
                    // for (int bit : hashes[i]) std::cout << bit;
                    // std::cout << ")" << std::endl;
                }
            }

            std::vector<int> query(const Eigen::VectorXd& queryPoint) {
                // std::cout << "[QUERY] Querying for point " << queryPoint.transpose() << std::endl;
                std::set<int> candidateIds;
                std::vector<HashKey> hashes = generateHash(queryPoint);

                for (int i = 0; i < numTables; i++) {
                    auto it = tables[i].find(hashes[i]);
                    if (it != tables[i].end()) {
                        // std::cout << "  [TABLE " << i << "] Bicket contains: ";
                        for (int id : it -> second) {
                            //std::cout << id << " ";
                            candidateIds.insert(id);
                        }
                        // std::cout << std::endl;
                    } else {
                        // std::cout << "  [TABLE " << i << "] Bucket is empty" << std::endl;
                    }
                }

                // std::cout << "[QUERY_RESULT] Candidates found: ";
                // for (int id : candidateIds) std::cout << id << " ";
                // std::cout << std::endl;

                return std::vector<int>(candidateIds.begin(), candidateIds.end());
            }
    };
}