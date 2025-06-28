#include <iostream>
#include <cuda_runtime.h>
#include "merlin_hashtable.cuh"
#include "benchmark_util.cuh"

using K = uint64_t;
using V = double;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;

void test_hybrid() {
    // Configuration parameters
    const size_t capacity = 10000000;  // 10M keys
    const size_t dim = 16;             // 8-dimensional vectors
    const size_t hbm_size = 1;      // 100MB HBM
    const size_t key_num_per_op = 1000000;  // 100K keys per operation

    // Initialize options
    TableOptions options;
    options.init_capacity = capacity;
    options.max_capacity = capacity;
    options.dim = dim;
    options.max_hbm_for_vectors = nv::merlin::GB(hbm_size);
    options.io_by_cpu = false;

    // Create hash table
    std::shared_ptr<Table> table = std::make_shared<Table>();
    table->init(options);

    // Allocate host memory
    K* h_keys;
    S* h_scores;
    V* h_vectors;
    bool* h_found;

    CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
    CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));
    CUDA_CHECK(cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * dim));
    CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

    // Allocate device memory
    K* d_keys;
    S* d_scores;
    V* d_vectors;
    bool* d_found;
    K* d_evict_keys;
    V* d_def_val;
    S* d_evict_scores;

    CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_scores, key_num_per_op * sizeof(S)));
    CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * dim));
    CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_evict_keys, key_num_per_op * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_def_val, key_num_per_op * sizeof(V) * dim));
    CUDA_CHECK(cudaMalloc(&d_evict_scores, key_num_per_op * sizeof(S)));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Test insert_and_evict operations
    std::cout << "Testing insert_and_evict..." << std::endl;
    for (int i = 0; i < 10; i++) {
        // Generate test data
        benchmark::create_continuous_keys<K, S>(h_keys, h_scores, key_num_per_op, i * key_num_per_op);
        benchmark::init_value_using_key<K, V>(h_keys, h_vectors, key_num_per_op, dim);

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, key_num_per_op * sizeof(V) * dim, cudaMemcpyHostToDevice));


        size_t* d_evicted_counter;
        CUDA_CHECK(cudaMalloc(&d_evicted_counter, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(d_evicted_counter, 0, sizeof(size_t)));

        // Execute insert_and_evict
        table->insert_and_evict(key_num_per_op, d_keys, d_vectors, nullptr,
                               d_evict_keys, d_def_val, nullptr, d_evicted_counter, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t evicted_count;
        CUDA_CHECK(cudaMemcpy(&evicted_count, d_evicted_counter, sizeof(size_t), cudaMemcpyDeviceToHost));
        std::cout << "Inserted batch " << i + 1 << " with " << evicted_count << " evicted keys" << std::endl;
    }

    // Test find operations
    std::cout << "\nTesting find..." << std::endl;
    for (int i = 0; i < 10; i++) {
        // Generate test data
        benchmark::create_continuous_keys<K, S>(h_keys, h_scores, key_num_per_op, i * key_num_per_op);

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S), cudaMemcpyHostToDevice));

        // Execute find
        table->find(key_num_per_op, d_keys, d_vectors, d_found, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Check results
        CUDA_CHECK(cudaMemcpy(h_found, d_found, key_num_per_op * sizeof(bool), cudaMemcpyDeviceToHost));
        int found_count = 0;
        for (int j = 0; j < key_num_per_op; j++) {
            if (h_found[j]) found_count++;
        }
        std::cout << "Batch " << i + 1 << " found: " << found_count << "/" << key_num_per_op << std::endl;
    }

    // Clean up resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_scores));
    CUDA_CHECK(cudaFreeHost(h_vectors));
    CUDA_CHECK(cudaFreeHost(h_found));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_evict_keys));
    CUDA_CHECK(cudaFree(d_def_val));
    CUDA_CHECK(cudaFree(d_evict_scores));
}

int main() {
    test_hybrid();
    return 0;
} 