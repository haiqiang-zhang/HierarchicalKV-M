/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "merlin_hashtable.cuh"
#include "nv/string_type.cuh"

using namespace nv::merlin;
using namespace nv;

// Define the key, value and score types
using K = uint64_t;  // Key type (integer)
using V = String;    // Value type (our custom string implementation)
using S = uint64_t;  // Score type (timestamp or other metric)

// Hashtable type with LRU eviction strategy
using StringHashTable = HashTable<K, V, S, EvictStrategy::kLru>;

// Helper function to generate random strings
std::string generateRandomString(std::mt19937& rng, size_t length) {
  const std::string charset = 
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::uniform_int_distribution<size_t> dist(0, charset.size() - 1);
  
  std::string result;
  result.reserve(length);
  
  for (size_t i = 0; i < length; ++i) {
    result += charset[dist(rng)];
  }
  
  return result;
}

int main() {
  // Initialize CUDA device
  int device_id = 0;
  cudaSetDevice(device_id);
  
  // Configure the hash table
  HashTableOptions options;
  options.init_capacity = 1000000;  // Initial capacity (1M)
  options.max_capacity = 1000000;   // Maximum capacity (1M)
  options.dim = 1;                  // Dimension of values (1 for strings)
  options.max_load_factor = 0.75;   // Load factor before rehashing
  options.max_hbm_for_vectors = GB(4);  // 4GB max HBM for values
  options.device_id = device_id;    // GPU device ID
  
  // Create and initialize the hash table
  StringHashTable table;
  table.init(options);
  
  std::cout << "Initialized StringHashTable with capacity: " << options.init_capacity << std::endl;
  
  // Parameters for the example
  const size_t num_items = 10000;  // Number of items to insert
  const size_t batch_size = 1000;  // Batch size for operations
  const size_t string_length = 32;  // Length of random strings
  
  // Random number generator
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  
  // Prepare host vectors for keys, values, and scores
  std::vector<K> h_keys(batch_size);
  std::vector<V> h_values(batch_size);
  std::vector<S> h_scores(batch_size);
  std::vector<bool> h_found(batch_size);
  
  // Allocate device memory
  K* d_keys = nullptr;
  V* d_values = nullptr;
  S* d_scores = nullptr;
  bool* d_found = nullptr;
  
  cudaMalloc(&d_keys, batch_size * sizeof(K));
  cudaMalloc(&d_values, batch_size * sizeof(V));
  cudaMalloc(&d_scores, batch_size * sizeof(S));
  cudaMalloc(&d_found, batch_size * sizeof(bool));
  
  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  std::cout << "Starting to insert " << num_items << " items..." << std::endl;
  
  // Insert items in batches
  for (size_t i = 0; i < num_items; i += batch_size) {
    size_t current_batch_size = std::min(batch_size, num_items - i);
    
    // Prepare batch data on host
    for (size_t j = 0; j < current_batch_size; ++j) {
      h_keys[j] = i + j;  // Sequential keys
      h_values[j] = generateRandomString(rng, string_length).c_str();  // Random string values
      h_scores[j] = static_cast<S>(i + j);  // Use key as score for simplicity
    }
    
    // Copy data to device
    cudaMemcpyAsync(d_keys, h_keys.data(), current_batch_size * sizeof(K), 
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, h_values.data(), current_batch_size * sizeof(V), 
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores, h_scores.data(), current_batch_size * sizeof(S), 
                   cudaMemcpyHostToDevice, stream);
    
    // Insert into the hash table
    table.insert_or_assign(current_batch_size, d_keys, d_values, d_scores, stream);
  }
  
  // Synchronize to ensure all inserts are complete
  cudaStreamSynchronize(stream);
  
  std::cout << "Inserted " << num_items << " items." << std::endl;
  std::cout << "Table size: " << table.size(stream) << std::endl;
  
  // Now let's retrieve some values
  std::cout << "Testing retrieval of items..." << std::endl;
  
  // Select random keys to retrieve
  std::uniform_int_distribution<K> key_dist(0, num_items - 1);
  
  for (size_t j = 0; j < batch_size; ++j) {
    h_keys[j] = key_dist(rng);  // Random keys
  }
  
  // Copy keys to device
  cudaMemcpyAsync(d_keys, h_keys.data(), batch_size * sizeof(K), 
                 cudaMemcpyHostToDevice, stream);
  
  // Find values in the hash table
  table.find(batch_size, d_keys, d_values, d_found, d_scores, stream);
  
  // Copy results back to host
  cudaMemcpyAsync(h_values.data(), d_values, batch_size * sizeof(V), 
                 cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_found.data(), d_found, batch_size * sizeof(bool), 
                 cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_scores.data(), d_scores, batch_size * sizeof(S), 
                 cudaMemcpyDeviceToHost, stream);
  
  // Synchronize to ensure all operations are complete
  cudaStreamSynchronize(stream);
  
  // Print some results
  size_t found_count = 0;
  
  for (size_t j = 0; j < 10 && j < batch_size; ++j) {  // Print first 10 results
    found_count += h_found[j] ? 1 : 0;
    
    std::cout << "Key: " << h_keys[j] 
              << ", Found: " << (h_found[j] ? "Yes" : "No");
    
    if (h_found[j]) {
      std::cout << ", Value: \"" << h_values[j].c_str() 
                << "\", Score: " << h_scores[j];
    }
    
    std::cout << std::endl;
  }
  
  std::cout << "Found " << found_count << " out of " << batch_size << " queried keys." << std::endl;
  
  // Clean up
  cudaStreamDestroy(stream);
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_scores);
  cudaFree(d_found);
  
  std::cout << "Example completed successfully." << std::endl;
  
  return 0;
} 