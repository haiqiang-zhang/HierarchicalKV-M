#pragma once

#include <string>
#include <cstdint>

// Flags structure definition for YCSB benchmark configuration
struct Flags {
  uint16_t gpu_id              = 0;
  uint64_t record_count        = 6ull * 1024 * 1024;    // preload rows
  uint64_t num_batch_ops       = 200;                    // total batched ops
  double   read_prop           = 1;                     // read fraction
  std::string distribution     = "zipf";
  double   theta               = 0;                     // zipf skew
  uint32_t multiget_batch_size = 64 * 1024;             // keys per kernel launch for multiget
  uint32_t multiset_batch_size = 64 * 1024;             // keys per kernel launch for multiset
  uint32_t dim                 = 5;                     // vector dimension
  uint64_t init_capacity       = 12ull * 1024 * 1024;   // slots
  uint32_t hbm_gb              = 18;                    // HBM for vectors
  uint64_t seed                = 42;
};


struct BenchmarkResult {
  double time_seconds;
  double ops_per_sec;
  double gbkv_per_sec;
};

// Function to parse command line arguments
Flags parse_flags(int argc, char** argv);

// Main YCSB benchmark function
BenchmarkResult run_ycsb(const Flags& cfg);