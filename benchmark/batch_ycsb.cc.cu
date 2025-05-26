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
#include <string>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "benchmark_util.cuh"    
#include "merlin_hashtable.cuh"  
#include "ycsb.cuh"

using namespace nv::merlin;
using benchmark::Timer;
using json = nlohmann::json;

// Custom parameters for batch benchmarking
struct BatchBenchmarkParams {
  std::vector<uint32_t> batch_sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152};
  std::vector<double> read_props    = {1.0};
  std::string output_file           = "batch_benchmark_results.json";
  uint16_t gpu_id                   = 0;
  uint64_t record_count             = 6ull * 1024 * 1024;    // preload rows
  uint64_t num_batch_ops            = 200;                    // total batched ops
  double   read_prop                = 1;                     // read fraction
  std::string distribution          = "zipf";
  double   theta                    = 0;                     // zipf skew
  uint32_t multiget_batch_size      = 64 * 1024;             // keys per kernel launch
  uint32_t multiset_batch_size      = 64 * 1024;             // keys per kernel launch
  uint32_t dim                      = 5;                     // vector dimension
  uint64_t init_capacity            = 12ull * 1024 * 1024;   // slots
  uint32_t hbm_gb                   = 18;                    // HBM for vectors
  uint64_t seed                     = 42;
};

static void usage(const char* prog, const bool is_error = false) {
  std::string usage_msg = "Usage: " + std::string(prog) + " [options]\n"
                        + "Options:\n"
                        + "  -h, --help             print this help message\n"
                        + "  --gpu_id=N             GPU ID (default 0)\n"
                        + "  --recordcount=N        preload N records (default 10M)\n"
                        + "  --num_batch_ops=N      execute N ops (default 200)\n"
                        + "  --distribution=d       zipf | uniform (default zipf)\n"
                        + "  --theta=f              zipf theta (default 0.99)\n"
                        + "  --dim=N                vector dimension (default 64)\n"
                        + "  --initcapacity=N       table capacity (default 64M)\n"
                        + "  --hbm_gb=N             HBM budget (default 16)\n"
                        + "  --seed=N               RNG seed (default 42)\n"
                        + "  --batch_sizes=L        comma-separated list of batch sizes (e.g. 1024,2048,4096)\n"
                        + "  --read_props=L         comma-separated list of read proportions (e.g. 0.0,0.5,1.0)\n"
                        + "  --output=s             output CSV file (default batch_benchmark_results.json)\n";
  if (is_error) {
    std::cerr << "Error: " << prog << " [options]\n"
              << usage_msg << std::endl;
  } else {
    std::cout << usage_msg << std::endl;
  }
}

// Helper function to parse comma-separated values
template<typename T>
std::vector<T> parse_comma_separated(const std::string& input) {
  std::vector<T> result;
  std::stringstream ss(input);
  std::string token;
  
  while (std::getline(ss, token, ',')) {
    std::stringstream tokenstream(token);
    T value;
    tokenstream >> value;
    result.push_back(value);
  }
  
  return result;
}

Flags parse_batch_flags(int argc, char** argv, BatchBenchmarkParams& batch_params) {
  Flags f;
  
  static struct option long_opts[] = {
      {"help",            no_argument,       nullptr, 'h'  },
      {"gpu_id",          required_argument, nullptr, 'i'  },
      {"recordcount",     required_argument, nullptr, 'r'  },
      {"num_batch_ops",   required_argument, nullptr, 'o'  },
      {"distribution",    required_argument, nullptr, 'd'  },
      {"theta",           required_argument, nullptr, 't'  },
      {"dim",             required_argument, nullptr, 'm'  },
      {"initcapacity",    required_argument, nullptr, 'c'  },
      {"hbm_gb",          required_argument, nullptr, 'g'  },
      {"seed",            required_argument, nullptr, 's'  },
      {"batch_sizes",     required_argument, nullptr, 'b'  },
      {"read_props",      required_argument, nullptr, 'p'  },
      {"output",          required_argument, nullptr, 'f'  },
      {0,0,0,0}
  };
  
  int opt;
  int long_idx;
  while ((opt = getopt_long(argc, argv, "h", long_opts, &long_idx)) != -1) {
    switch (opt) {
      case 'h': usage(argv[0], false); exit(EXIT_SUCCESS);
      case 'i': batch_params.gpu_id          = std::strtoul (optarg, nullptr, 10); break;
      case 'r': batch_params.record_count    = std::strtoull(optarg, nullptr, 10);   break;
      case 'o': batch_params.num_batch_ops = std::strtoull(optarg, nullptr, 10);   break;
      case 'd': batch_params.distribution    = optarg;                               break;
      case 't': batch_params.theta           = std::strtod (optarg, nullptr);        break;
      case 'm': batch_params.dim             = std::strtoul (optarg, nullptr, 10);   break;
      case 'c': batch_params.init_capacity   = std::strtoull(optarg, nullptr, 10);   break;
      case 'g': batch_params.hbm_gb          = std::strtoul (optarg, nullptr, 10);   break;
      case 's': batch_params.seed            = std::strtoull(optarg, nullptr, 10);   break;
      case 'b': batch_params.batch_sizes = parse_comma_separated<uint32_t>(optarg); break;
      case 'p': batch_params.read_props = parse_comma_separated<double>(optarg);    break;
      case 'f': batch_params.output_file = optarg;                        break;
      default : usage(argv[0], true); exit(EXIT_FAILURE);
    }
  }


  // convert batch_params to Flags
  f.gpu_id = batch_params.gpu_id;
  f.record_count = batch_params.record_count;
  f.num_batch_ops = batch_params.num_batch_ops;
  f.read_prop = batch_params.read_prop;
  f.distribution = batch_params.distribution;
  f.theta = batch_params.theta;
  f.dim = batch_params.dim;
  f.init_capacity = batch_params.init_capacity;
  f.hbm_gb = batch_params.hbm_gb;
  f.seed = batch_params.seed;
  
  return f;
}

// Run a batch of benchmarks with varying parameters
void run_batch_benchmark(const Flags& base_cfg, const BatchBenchmarkParams& batch_params) {
  std::ofstream result_file(batch_params.output_file);
  if (!result_file.is_open()) {
    std::cerr << "Failed to open output file: " << batch_params.output_file << std::endl;
    return;
  }
  
  json results = json::array();
  std::cout << "Starting batch benchmark..." << std::endl;
  
  // Run all combinations of batch sizes and read proportions
  for (const auto& batch_size : batch_params.batch_sizes) {
    for (const auto& read_prop : batch_params.read_props) {
      Flags cfg = base_cfg;
      cfg.multiget_batch_size = batch_size;
      cfg.multiset_batch_size = batch_size;
      cfg.read_prop = read_prop;
      
      std::cout << "Running benchmark with batch_size=" << batch_size 
                << ", read_prop=" << read_prop << std::endl;
      
      BenchmarkResult result = run_ycsb(cfg);
      std::array<int, 1> gpu_ids = {cfg.gpu_id};
      json benchmark_result = {
        {"workload", "YCSB_Batch"},
        {"binding", "hierarchical_kv"},
        {"distribution", cfg.distribution},
        {"num_records", cfg.record_count},
        {"batch_size", batch_size},
        {"num_batch_ops", cfg.num_batch_ops},
        {"multiget_prob", read_prop},
        {"multiget_batch_size", cfg.multiget_batch_size},
        {"multiset_batch_size", cfg.multiset_batch_size},
        {"total_time", result.time_seconds},
        {"throughput", result.ops_per_sec},
        {"gbkv_per_sec", result.gbkv_per_sec},
        {"gpu_device", gpu_ids},
        {"min_field_length", 5},
        {"max_field_length", 5},
        {"field_count", cfg.dim},
        {"init_capacity", cfg.init_capacity},
        {"hbm_gb", cfg.hbm_gb},
        {"load_factor", cfg.record_count / (double) cfg.init_capacity},
        {"seed", cfg.seed}
      };
      
      if (cfg.distribution == "zipf") {
        benchmark_result["zipfian_theta"] = cfg.theta;
      }
      
      results.push_back(benchmark_result);
      std::cout << "  Completed: " << result.ops_per_sec << " ops/sec" << std::endl;
    }
  }
  
  result_file << std::setw(4) << results << std::endl;
  result_file.close();
  std::cout << "All benchmarks completed. Results saved to " << batch_params.output_file << std::endl;
}

int main(int argc, char** argv) {
  BatchBenchmarkParams batch_params;
  Flags base_cfg = parse_batch_flags(argc, argv, batch_params);
  
  // Display configuration
  std::cout << "Batch YCSB Benchmark Configuration:" << std::endl;
  std::cout << "  GPU ID: " << base_cfg.gpu_id << std::endl;
  std::cout << "  Record count: " << base_cfg.record_count << std::endl;
  std::cout << "  Number of batch operations: " << base_cfg.num_batch_ops << std::endl;
  std::cout << "  Distribution: " << base_cfg.distribution << std::endl;
  if (base_cfg.distribution == "zipf") {
    std::cout << "  Zipf theta: " << base_cfg.theta << std::endl;
  }
  std::cout << "  Vector dimension: " << base_cfg.dim << std::endl;
  std::cout << "  Initial capacity: " << base_cfg.init_capacity << std::endl;
  std::cout << "  Load factor: " << base_cfg.record_count / (double) base_cfg.init_capacity << std::endl;
  std::cout << "  HBM size (GB): " << base_cfg.hbm_gb << std::endl;
  std::cout << "  Seed: " << base_cfg.seed << std::endl;
  
  std::cout << "  Batch sizes to test: ";
  for (size_t i = 0; i < batch_params.batch_sizes.size(); i++) {
    std::cout << batch_params.batch_sizes[i];
    if (i < batch_params.batch_sizes.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
  
  std::cout << "  Read proportions to test: ";
  for (size_t i = 0; i < batch_params.read_props.size(); i++) {
    std::cout << batch_params.read_props[i];
    if (i < batch_params.read_props.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
  
  std::cout << "  Output file: " << batch_params.output_file << std::endl;
  
  // Run the batch benchmark
  run_batch_benchmark(base_cfg, batch_params);
  
  return 0;
} 