#include "ycsb.cuh"
#include <iostream>

int main(int argc, char** argv) {
  Flags cfg = parse_flags(argc, argv);
  BenchmarkResult result = run_ycsb(cfg);
  
  // Print results summary
  std::cout << "\nBenchmark Results Summary:" << std::endl;
  std::cout << "  Total time: " << result.time_seconds << " seconds" << std::endl;
  std::cout << "  Operations per second: " << result.ops_per_sec << std::endl;
  std::cout << "  GB-KV/s: " << result.gbkv_per_sec << std::endl;
  
  // Check if benchmark ran successfully (time > 0)
  return (result.time_seconds > 0) ? 0 : EXIT_FAILURE;
}
