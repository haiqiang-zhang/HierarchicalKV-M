#include <cuda_runtime.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <thread>
#include <numeric>
#include <string>
#include <functional>
#include <map>

#include "benchmark_util.cuh"    
#include "merlin_hashtable.cuh"  
#include "ycsb.cuh"

using namespace nv::merlin;
using benchmark::Timer;



// Custom string type that is compatible with CUDA and Merlin library
struct CustomString {
  char data[MAX_STRING_LENGTH];
  
  __host__ __device__ CustomString() {
    memset(data, 0, MAX_STRING_LENGTH);
  }
  
  __host__ __device__ CustomString(const CustomString& other) {
    memcpy(data, other.data, MAX_STRING_LENGTH);
  }
  
  __host__ __device__ CustomString& operator=(const CustomString& other) {
    if (this != &other) {
      memcpy(data, other.data, MAX_STRING_LENGTH);
    }
    return *this;
  }
  
  __host__ __device__ CustomString& operator=(const char* str) {
    if (str) {
      size_t len = 0;
      while (str[len] && len < MAX_STRING_LENGTH - 1) len++;
      memcpy(data, str, len);
      data[len] = '\0';
    } else {
      data[0] = '\0';
    }
    return *this;
  }
  
  __host__ __device__ CustomString& operator+=(const CustomString& other) {
    size_t len = 0;
    while (data[len] && len < MAX_STRING_LENGTH - 1) len++;
    
    size_t i = 0;
    while (other.data[i] && len + i < MAX_STRING_LENGTH - 1) {
      data[len + i] = other.data[i];
      i++;
    }
    data[len + i] = '\0';
    return *this;
  }
  
  __host__ __device__ char& operator[](size_t idx) {
    return data[idx];
  }
  
  __host__ __device__ const char& operator[](size_t idx) const {
    return data[idx];
  }
};

static void usage(const char* prog, const bool is_error = false) {
  std::string usage_msg = "Usage: " + std::string(prog) + " [options]\n"
                         + "Options:\n"
                         + "  -h, --help             print this help message\n"
                         + "  --gpu_id=N             GPU ID (default 0)\n"
                         + "  --recordcount=N        preload N records (default 6M)\n"
                         + "  --num_batch_ops=N      number of batch operations to execute (default 10)\n"
                         + "  --readproportion=f     read ratio 0‑1 (default 1.0)\n"
                         + "  --distribution=d       zipf | uniform (default zipf)\n"
                         + "  --theta=f              zipf theta (default 0.0)\n"
                         + "  --multiget_batch=N     keys per read batch (default 64K)\n"
                         + "  --multiset_batch=N     keys per write batch (default 64K)\n"
                         + "  --dim=N                vector dimension (default 5)\n"
                         + "  --initcapacity=N       table capacity (default 12M)\n"
                         + "  --hbm_gb=N             HBM budget in GB (default 18)\n"
                         + "  --seed=N               RNG seed (default 42)\n"
                         + "  --num_streams=N        Number of CUDA streams (default 1)\n";
  if (is_error) {
    std::cerr << "Error: " << prog << " [options]\n"
              << usage_msg << std::endl;
  } else {
    std::cout << usage_msg << std::endl;
  }
}

Flags parse_flags(int argc, char** argv) {
  Flags f;
  static struct option long_opts[] = {
      {"help",            no_argument,       nullptr, 'h'  },
      {"gpu_id",          required_argument, nullptr, 'i'  },
      {"recordcount",     required_argument, nullptr, 'r'  },
      {"num_batch_ops",   required_argument, nullptr, 'o'  },
      {"readproportion",  required_argument, nullptr, 'p'  },
      {"distribution",    required_argument, nullptr, 'd'  },
      {"theta",           required_argument, nullptr, 't'  },
      {"multiget_batch",  required_argument, nullptr, 'g'  },
      {"multiset_batch",  required_argument, nullptr, 's'  },
      {"dim",             required_argument, nullptr, 'm'  },
      {"initcapacity",    required_argument, nullptr, 'c'  },
      {"hbm_gb",          required_argument, nullptr, 'b'  },
      {"seed",            required_argument, nullptr, 'e'  },
      {"num_streams",     required_argument, nullptr, 'N'  },
      {0,0,0,0}
  };
  int opt;
  int long_idx;
  while ((opt = getopt_long(argc, argv, "h", long_opts, &long_idx)) != -1) {
    switch (opt) {
      case 'h': usage(argv[0], false); exit(EXIT_SUCCESS);
      case 'i': f.gpu_id          = std::strtoul (optarg, nullptr, 10); break;
      case 'r': f.record_count    = std::strtoull(optarg, nullptr, 10);   break;
      case 'o': f.num_batch_ops   = std::strtoull(optarg, nullptr, 10);   break;
      case 'p': f.read_prop       = std::strtod (optarg, nullptr);        break;
      case 'd': f.distribution    = optarg;                               break;
      case 't': f.theta           = std::strtod (optarg, nullptr);        break;
      case 'g': f.multiget_batch_size = std::strtoul (optarg, nullptr, 10);   break;
      case 's': f.multiset_batch_size = std::strtoul (optarg, nullptr, 10);   break;
      case 'm': f.dim             = std::strtoul (optarg, nullptr, 10);   break;
      case 'c': f.init_capacity   = std::strtoull(optarg, nullptr, 10);   break;
      case 'b': f.hbm_gb          = std::strtoul (optarg, nullptr, 10);   break;
      case 'e': f.seed            = std::strtoull(optarg, nullptr, 10);   break;
      case 'N': f.num_streams     = std::strtoul (optarg, nullptr, 10);   break;
      default : usage(argv[0], true); exit(EXIT_FAILURE);
    }
  }
  if (f.read_prop < 0.0 || f.read_prop > 1.0) {
    std::cerr << "readproportion must be in [0,1]\n"; exit(EXIT_FAILURE);
  }
  return f;
}

// -------------------------------------------------------------
// Key generators
// -------------------------------------------------------------
class KeyGenerator {
 public:
  virtual uint64_t operator()() = 0;
  virtual std::unique_ptr<KeyGenerator> clone() const = 0; 
  virtual ~KeyGenerator() = default;
};

class UniformKeyGen : public KeyGenerator {
  std::uniform_int_distribution<uint64_t> dist_;
  std::mt19937_64 eng_;
 public:
  UniformKeyGen(uint64_t max_key, uint64_t seed): dist_(0, max_key-1), eng_(seed) {}
  uint64_t operator()() override { return dist_(eng_); }

  std::unique_ptr<KeyGenerator> clone() const override {
    return std::make_unique<UniformKeyGen>(*this);
  }
};

// Approximate Zipf generator using rejection sampling.
class ZipfKeyGen : public KeyGenerator {
  uint64_t n_;              // number of items
  uint64_t base_;           // min value
  double theta_;            // zipfian constant
  double alpha_;            // computed from theta
  double zeta_n_;          // zeta(n)
  double zeta2_theta_;     // zeta(2,theta)
  double eta_;             // computed from theta
  std::mt19937_64 eng_;    // random number generator
  std::uniform_real_distribution<double> uni_;

  double zeta(uint64_t n, double theta, uint64_t start = 0, double initialSum = 0) {
    double sum = initialSum;
    for(uint64_t i = start; i < n; i++) {
      sum += 1.0 / std::pow(i + 1, theta);
    }
    return sum;
  }

 public:
  ZipfKeyGen(uint64_t n, double theta, uint64_t seed)
    : n_(n), base_(0), theta_(theta), eng_(seed), uni_(0.0, 1.0) {
    // Compute zeta values
    zeta_n_ = zeta(n_, theta_);
    zeta2_theta_ = zeta(2, theta_);
    alpha_ = 1.0 / (1.0 - theta_);
    eta_ = (1.0 - std::pow(2.0/n_, 1.0 - theta_)) / (1.0 - zeta2_theta_ / zeta_n_);
  }

  uint64_t operator()() override {
    double u = uni_(eng_);
    double uz = u * zeta_n_;

    if (uz < 1.0) return base_;
    if (uz < 1.0 + std::pow(0.5, theta_)) return base_ + 1;

    uint64_t ret = base_ + static_cast<uint64_t>(
      n_ * std::pow(eta_ * u - eta_ + 1, alpha_)
    );
    return std::min(ret, n_ - 1);  // ensure we don't exceed range
  }

  std::unique_ptr<KeyGenerator> clone() const override {
    return std::make_unique<ZipfKeyGen>(*this);
  }
};

using K = uint64_t;
using S = uint64_t;
using V = CustomString;
using HKVTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
using TableOptions = nv::merlin::HashTableOptions;


struct DeviceBuffers {
  K* d_keys;
  V* d_values;
  V* d_values_out;
  S* d_scores;
  bool* d_found;
  V** d_values_ptr;

  explicit DeviceBuffers(uint32_t batch_size, uint32_t dim) {
    CUDA_CHECK(cudaMalloc(&d_keys,        batch_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values,      batch_size * dim * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_values_out,  batch_size * dim * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_scores,      batch_size * sizeof(S)));
    CUDA_CHECK(cudaMalloc(&d_found,       batch_size * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_values_ptr,  batch_size * sizeof(V*)));
  }
  ~DeviceBuffers() {
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_values_out);
    cudaFree(d_scores);
    cudaFree(d_found);
    cudaFree(d_values_ptr);
  }
};


void generate_keys_parallel(uint32_t                 bs,
                            const KeyGenerator&      keygen_proto,
                            std::vector<K>&          out_keys,
                            unsigned                 T = std::thread::hardware_concurrency())
{
    // print the number of threads
    // std::cout << "T: " << T << std::endl;
    std::vector<std::vector<K>> local_keys(T);   

    const uint32_t chunk = (bs + T - 1) / T;     
    auto worker = [&](unsigned tid) {
        uint32_t begin = tid * chunk;
        uint32_t end   = std::min(begin + chunk, bs);
        if (begin >= end) return;               

        auto kg = keygen_proto.clone();        

        local_keys[tid].reserve(end - begin);
        for (uint32_t i = begin; i < end; ++i) {
            local_keys[tid].push_back((*kg)()); 
        }
    };

    // Start threads
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < T; ++t)
        pool.emplace_back(worker, t);
    for (auto& th : pool)
        th.join();                 

    out_keys.clear();
    size_t total = 0;
    for (const auto& v : local_keys) total += v.size();
    out_keys.reserve(total);

    for (auto& v : local_keys) {
        out_keys.insert(out_keys.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
    }
}


BenchmarkResult run_ycsb(const Flags& cfg) {
  // Set the GPU device first
  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  std::cout << "Available CUDA devices: " << device_count << std::endl;

  std::cout << "GPU: " << cfg.gpu_id << std::endl;
  // ----- Create key generator -----
  std::unique_ptr<KeyGenerator> keygen;
  if (cfg.distribution == "uniform") {
    keygen = std::make_unique<UniformKeyGen>(cfg.record_count, cfg.seed);
  } else if (cfg.distribution == "zipf") {
    keygen = std::make_unique<ZipfKeyGen>(cfg.record_count, cfg.theta, cfg.seed);
  } else {
    std::cerr << "Unsupported distribution: " << cfg.distribution << std::endl;
    return {0.0, 0.0, 0.0};
  }

  // ----- Initialize table -----
  TableOptions options;
  options.init_capacity = cfg.init_capacity;
  options.max_capacity  = cfg.init_capacity;
  options.dim           = cfg.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(cfg.hbm_gb);
  options.io_by_cpu     = false;
  options.device_id     = cfg.gpu_id;  // Set the device ID in options

  std::unique_ptr<HKVTable> table = std::make_unique<HKVTable>();
  table->init(options);

  // ----- Allocate host buffers -----
  std::vector<K> h_keys(cfg.multiset_batch_size); // Use multiset_batch_size for write operations
  std::vector<CustomString> h_vals(cfg.multiset_batch_size * cfg.dim);
  std::mt19937 rng(cfg.seed); 
  
  // Host storage for preloaded data for verification
  std::map<K, std::vector<CustomString>> preloaded_data_host;

  // Characters for random string generation
  const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::uniform_int_distribution<size_t> char_dist(0, charset.size() - 1);
  
  // ----- Device buffers -----
  // Ensure dbuf is large enough for both read and write batches in the workload phase
  uint32_t max_op_batch_size = std::max(cfg.multiget_batch_size, cfg.multiset_batch_size);

  std::vector<DeviceBuffers> stream_specific_dbufs;
  for (uint32_t i = 0; i < cfg.num_streams; ++i) {
    stream_specific_dbufs.emplace_back(max_op_batch_size, cfg.dim);
  }
  std::cout << "Created " << cfg.num_streams << " stream-specific device buffers." << std::endl;

  // Create a vector of CUDA streams
  std::vector<cudaStream_t> streams(cfg.num_streams);
  for (uint32_t i = 0; i < cfg.num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }
  cudaStream_t default_stream_for_preload = streams[0]; 

  // ------------------------------------------------------
  // Preload phase (multiset / insert_or_assign)
  // ------------------------------------------------------
  std::cout << "Preloading " << cfg.record_count << " records..." << std::endl;
  uint64_t inserted = 0;
  while (inserted < cfg.record_count) {
    uint32_t this_batch = std::min<uint64_t>(cfg.multiset_batch_size, cfg.record_count - inserted);
    // Generate keys
    for(uint32_t i=0; i<this_batch; ++i) h_keys[i] = inserted + i;
    
    // Generate values
    for(uint32_t i=0; i<this_batch*cfg.dim; ++i) {
      CustomString& str = h_vals[i];
      memset(str.data, 0, MAX_STRING_LENGTH);
      
      for (uint32_t j = 0; j < MAX_STRING_LENGTH-1; ++j) {
        str[j] = charset[char_dist(rng)];
      }

      str[MAX_STRING_LENGTH-1] = '\0';
    }
    
    for (uint32_t k_idx = 0; k_idx < this_batch; ++k_idx) {
        K current_key = h_keys[k_idx];
        std::vector<CustomString> vals_for_key(cfg.dim);
        for (uint32_t d = 0; d < cfg.dim; ++d) {
            vals_for_key[d] = h_vals[k_idx * cfg.dim + d];
        }
        preloaded_data_host[current_key] = vals_for_key;
    }

    CUDA_CHECK(cudaMemcpy(stream_specific_dbufs[0].d_keys, h_keys.data(), this_batch*sizeof(K), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(stream_specific_dbufs[0].d_values, h_vals.data(), this_batch*cfg.dim*sizeof(V), cudaMemcpyHostToDevice));
    table->insert_or_assign(this_batch, stream_specific_dbufs[0].d_keys, stream_specific_dbufs[0].d_values, nullptr, default_stream_for_preload);
    inserted += this_batch;
    if (inserted % (1024ull * 1024ull) == 0) {
      std::cout << "Preloaded " << inserted << " records..." << std::endl;
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(default_stream_for_preload));

  // ------------------------------------------------------
  // Workload phase
  // ------------------------------------------------------
  std::cout << "Executing " << cfg.num_batch_ops << " batch operations..." << std::endl;
  double total_time = 0;
  Timer<double> timer;
  uint64_t total_ops = 0;

  // Create a vector to determine the type of each batch (true for read, false for write)
  std::vector<bool> batch_types(cfg.num_batch_ops);
  uint64_t num_read_batches = static_cast<uint64_t>(cfg.num_batch_ops * cfg.read_prop);
  std::fill(batch_types.begin(), batch_types.begin() + num_read_batches, true);
  std::fill(batch_types.begin() + num_read_batches, batch_types.end(), false);
  
  // Shuffle the batch types
  std::mt19937_64 rng_batch(cfg.seed + 1); 
  std::shuffle(batch_types.begin(), batch_types.end(), rng_batch);

  // --- Pre-generate all keys, values, and workload operations ---
  std::cout << "Pre-generating all keys, values, and operations for " << cfg.num_batch_ops << " batches..." << std::endl;
  std::vector<std::vector<K>> all_batch_keys(cfg.num_batch_ops);
  std::vector<std::vector<CustomString>> all_batch_values(cfg.num_batch_ops); 
  std::vector<std::function<void()>> workload_batch_fns;
  workload_batch_fns.reserve(cfg.num_batch_ops);

  // GPU storage for all read results
  std::vector<V*> d_all_results_values_out_list; 
  std::vector<bool*> d_all_results_found_list;    
  std::vector<uint32_t> read_op_batch_indices;    
  std::vector<uint32_t> read_op_actual_sizes;   

  uint32_t current_stream_idx = 0; // Initialize stream index for round-robin assignment

  for (uint64_t batch_idx = 0; batch_idx < cfg.num_batch_ops; ++batch_idx) {
    bool is_read_op = batch_types[batch_idx];
    uint32_t current_batch_size = is_read_op ? cfg.multiget_batch_size : cfg.multiset_batch_size;
    total_ops += current_batch_size; // Calculate total_ops here, outside the timed section
    

    generate_keys_parallel(current_batch_size, *keygen, all_batch_keys[batch_idx]);

    if (!is_read_op) { // It's a write batch, generate values
        all_batch_values[batch_idx].resize(current_batch_size * cfg.dim);
        for (uint32_t i = 0; i < current_batch_size * cfg.dim; ++i) {
            CustomString& str = all_batch_values[batch_idx][i];
            for (uint32_t j = 0; j < MAX_STRING_LENGTH - 1; ++j) {
                str[j] = charset[char_dist(rng)];
            }
            str[MAX_STRING_LENGTH - 1] = '\0';
        }
    }

    // Create and store the lambda for the current batch operation
    cudaStream_t captured_stream = streams[current_stream_idx];
    DeviceBuffers& captured_dbuf = stream_specific_dbufs[current_stream_idx];

    if (is_read_op) {
        // Allocate GPU memory for this specific read batch's results
        V* d_current_batch_values_out;
        bool* d_current_batch_found;
        CUDA_CHECK(cudaMalloc(&d_current_batch_values_out, current_batch_size * cfg.dim * sizeof(V)));
        CUDA_CHECK(cudaMalloc(&d_current_batch_found, current_batch_size * sizeof(bool)));
        
        d_all_results_values_out_list.push_back(d_current_batch_values_out);
        d_all_results_found_list.push_back(d_current_batch_found);
        read_op_batch_indices.push_back(batch_idx);
        read_op_actual_sizes.push_back(current_batch_size);

        workload_batch_fns.emplace_back(
            [&keys = all_batch_keys[batch_idx], 
             table_ptr = table.get(),
             stream = captured_stream,
             &dbuf_ref = captured_dbuf, // Stream-specific working buffer
             d_target_values_out = d_current_batch_values_out, // Destination for this batch's values
             d_target_found = d_current_batch_found]           // Destination for this batch's found flags
            () { 
                CUDA_CHECK(cudaMemcpyAsync(dbuf_ref.d_keys, keys.data(), keys.size()*sizeof(K), cudaMemcpyHostToDevice, stream));
                table_ptr->find(keys.size(), dbuf_ref.d_keys, d_target_values_out, d_target_found, nullptr, stream);
            }
        );
    } else { // Write op
        workload_batch_fns.emplace_back(
            [&keys = all_batch_keys[batch_idx], 
             &vals = all_batch_values[batch_idx], 
             dim_val = cfg.dim, 
             table_ptr = table.get(),
             stream = captured_stream,
             &dbuf_ref = captured_dbuf] // Stream-specific working buffer
            () { 
                CUDA_CHECK(cudaMemcpyAsync(dbuf_ref.d_keys, keys.data(), keys.size()*sizeof(K), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dbuf_ref.d_values, vals.data(), keys.size()*dim_val*sizeof(V), cudaMemcpyHostToDevice, stream));
                table_ptr->insert_or_assign(keys.size(), dbuf_ref.d_keys, dbuf_ref.d_values, nullptr, stream);
            }
        );
    }
    current_stream_idx = (current_stream_idx + 1) % cfg.num_streams;
  }
  std::cout << "Pre-generation complete." << std::endl;
  
  // --- Start Timer for Workload ---
  timer.start();
  

  for (uint64_t batch_op_idx = 0; batch_op_idx < cfg.num_batch_ops; ++batch_op_idx) {
    workload_batch_fns[batch_op_idx]();
  }
  
  // --- Synchronize and Stop Timer ---
  for (uint32_t i = 0; i < cfg.num_streams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
  timer.end();
  total_time = timer.getResult();
  double secs = total_time;
  double ops_per_sec = total_ops / secs;
  double gbkv_per_sec = total_ops / secs / 1e9;

  std::cout << "total_ops,time_ms,ops_per_sec,GB-kv/s\n";
  std::cout << total_ops << "," << secs*1000.0 << "," << ops_per_sec << "," << gbkv_per_sec << std::endl;

  // --- Verification Phase ---
  // std::cout << "Copying results from GPU to Host for verification..." << std::endl;
  // std::vector<std::vector<CustomString>> host_all_read_values_out(d_all_results_values_out_list.size());
  // std::vector<std::vector<unsigned char>> host_all_read_found_uchar(d_all_results_found_list.size());

  // for (size_t i = 0; i < d_all_results_values_out_list.size(); ++i) {
  //     uint32_t num_keys_in_batch = read_op_actual_sizes[i];

  //     host_all_read_values_out[i].resize(num_keys_in_batch * cfg.dim);
  //     host_all_read_found_uchar[i].resize(num_keys_in_batch);

  //     CUDA_CHECK(cudaMemcpy(host_all_read_values_out[i].data(),
  //                           d_all_results_values_out_list[i],
  //                           num_keys_in_batch * cfg.dim * sizeof(V),
  //                           cudaMemcpyDeviceToHost));
  //     CUDA_CHECK(cudaMemcpy(host_all_read_found_uchar[i].data(),
  //                           d_all_results_found_list[i],
  //                           num_keys_in_batch * sizeof(bool),
  //                           cudaMemcpyDeviceToHost));
  // }
  // std::cout << "Copying complete. Starting verification..." << std::endl;
  
  // uint64_t correct_reads_found = 0;
  // uint64_t correct_reads_not_found = 0;
  // uint64_t incorrect_reads_value_mismatch = 0;
  // uint64_t incorrect_reads_should_be_found = 0;
  // uint64_t incorrect_reads_should_not_be_found = 0;
  // uint64_t total_read_keys_verified = 0;

  // for (size_t i = 0; i < host_all_read_values_out.size(); ++i) { // Loop through read batches
  //     uint32_t original_batch_idx = read_op_batch_indices[i];
  //     const std::vector<K>& current_read_op_keys = all_batch_keys[original_batch_idx];
  //     const std::vector<CustomString>& actual_values_out_flat = host_all_read_values_out[i];
  //     const std::vector<unsigned char>& actual_found_flags_uchar = host_all_read_found_uchar[i];

  //     for (size_t k_idx = 0; k_idx < current_read_op_keys.size(); ++k_idx) { // Loop through keys in batch
  //         total_read_keys_verified++;
  //         K key_queried = current_read_op_keys[k_idx];
  //         bool found_in_table = (actual_found_flags_uchar[k_idx] != 0);

  //         auto it_preloaded = preloaded_data_host.find(key_queried);
  //         bool was_preloaded = (it_preloaded != preloaded_data_host.end());

  //         if (found_in_table) {
  //             if (was_preloaded) {
  //                 bool match = true;
  //                 const std::vector<CustomString>& expected_vals_for_key = it_preloaded->second;
  //                 for (uint32_t d = 0; d < cfg.dim; ++d) {
  //                     if (memcmp(actual_values_out_flat[k_idx * cfg.dim + d].data, expected_vals_for_key[d].data, MAX_STRING_LENGTH) != 0) {
  //                         match = false;
  //                         break;
  //                     }
  //                 }
  //                 if (match) {
  //                     correct_reads_found++;
  //                 } else {
  //                     incorrect_reads_value_mismatch++;
  //                     // Optional: Print mismatch details
  //                     // std::cerr << "Mismatch for key " << key_queried << std::endl;
  //                 }
  //             } else {
  //                 incorrect_reads_should_not_be_found++;
  //                 // std::cerr << "Key " << key_queried << " found but was not preloaded." << std::endl;
  //             }
  //         } else { // Not found in table
  //             if (was_preloaded) {
  //                 incorrect_reads_should_be_found++;
  //                 // std::cerr << "Key " << key_queried << " was preloaded but not found." << std::endl;
  //             } else {
  //                 correct_reads_not_found++;
  //             }
  //         }
  //     }
  // }
  // std::cout << "Verification Complete:" << std::endl;
  // std::cout << "  Total read keys verified: " << total_read_keys_verified << std::endl;
  // std::cout << "  Correct reads found: " << correct_reads_found << std::endl;
  // std::cout << "  Correct reads not found: " << correct_reads_not_found << std::endl;
  // std::cout << "  Incorrect - Value Mismatch: " << incorrect_reads_value_mismatch << std::endl;
  // std::cout << "  Incorrect - Should Have Been Found (but wasn't): " << incorrect_reads_should_be_found << std::endl;
  // std::cout << "  Incorrect - Should Not Have Been Found (but was): " << incorrect_reads_should_not_be_found << std::endl;


  // // Clean up GPU memory for results
  // for (V* ptr : d_all_results_values_out_list) {
  //     if (ptr) CUDA_CHECK(cudaFree(static_cast<void*>(ptr)));
  // }
  // for (bool* ptr : d_all_results_found_list) {
  //     if (ptr) CUDA_CHECK(cudaFree(static_cast<void*>(ptr)));
  // }
  // d_all_results_values_out_list.clear();
  // d_all_results_found_list.clear();

  stream_specific_dbufs.clear();
 
  for (uint32_t i = 0; i < cfg.num_streams; ++i) {

    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }


  return {secs, ops_per_sec, gbkv_per_sec};
}