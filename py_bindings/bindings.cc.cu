#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

#include "merlin_hashtable.cuh"
#include "mlkv/string_type.cuh"

using namespace nv::merlin;
namespace py = pybind11;

PYBIND11_MODULE(merlin_hashtable_python, m) {
    using HashTableOptionsType = nv::merlin::HashTableOptions;
    using MemoryPoolOptionsType = nv::merlin::MemoryPoolOptions;

    /* ─────────────────────  MemoryPoolOptions  ───────────────────── */
    py::class_<MemoryPoolOptionsType>(m, "MemoryPoolOptions", py::module_local(),
        "Settings for the buffer-pool allocator.")
        .def(py::init<>())                       // uses the C++ defaults
        .def_readwrite("max_stock",   &MemoryPoolOptionsType::max_stock,
            "Number of buffers kept in reserve.")
        .def_readwrite("max_pending", &MemoryPoolOptionsType::max_pending,
            "Maximum number of awaitable buffers before threads block.")
        .def("__repr__", [](const MemoryPoolOptionsType& o) {
            return "<MemoryPoolOptions max_stock="  + std::to_string(o.max_stock) +
                   " max_pending=" + std::to_string(o.max_pending) + ">";
        });

    /* ─────────────────────  HashTableOptions  ───────────────────── */
    py::class_<HashTableOptionsType>(m, "HashTableOptions", py::module_local(),
        "Configuration for the GPU hash table.")
        .def(py::init<>())                       // keeps all C++ defaults
        .def_readwrite("init_capacity",            &HashTableOptionsType::init_capacity)
        .def_readwrite("max_capacity",             &HashTableOptionsType::max_capacity)
        .def_readwrite("max_hbm_for_vectors",      &HashTableOptionsType::max_hbm_for_vectors)
        .def_readwrite("max_bucket_size",          &HashTableOptionsType::max_bucket_size)
        .def_readwrite("dim",                      &HashTableOptionsType::dim)
        .def_readwrite("max_load_factor",          &HashTableOptionsType::max_load_factor)
        .def_readwrite("block_size",               &HashTableOptionsType::block_size)
        .def_readwrite("io_block_size",            &HashTableOptionsType::io_block_size)
        .def_readwrite("device_id",                &HashTableOptionsType::device_id)
        .def_readwrite("io_by_cpu",                &HashTableOptionsType::io_by_cpu)
        .def_readwrite("use_constant_memory",      &HashTableOptionsType::use_constant_memory)
        .def_readwrite("reserved_key_start_bit",   &HashTableOptionsType::reserved_key_start_bit)
        .def_readwrite("num_of_buckets_per_alloc", &HashTableOptionsType::num_of_buckets_per_alloc)
        .def_readwrite("device_memory_pool",       &HashTableOptionsType::device_memory_pool)
        .def_readwrite("host_memory_pool",         &HashTableOptionsType::host_memory_pool)
        .def("__repr__", [](const HashTableOptionsType& o) {
            return "<HashTableOptions init_capacity=" + std::to_string(o.init_capacity) +
                   " max_capacity=" + std::to_string(o.max_capacity) +
                   " max_hbm_for_vectors=" + std::to_string(o.max_hbm_for_vectors) +
                   " max_bucket_size=" + std::to_string(o.max_bucket_size) +
                   " dim=" + std::to_string(o.dim) +
                   " max_load_factor=" + std::to_string(o.max_load_factor) +
                   " block_size=" + std::to_string(o.block_size) +
                   " io_block_size=" + std::to_string(o.io_block_size) +
                   " device_id=" + std::to_string(o.device_id) +
                   " io_by_cpu=" + std::to_string(o.io_by_cpu) +
                   " use_constant_memory=" + std::to_string(o.use_constant_memory) +
                   " reserved_key_start_bit=" + std::to_string(o.reserved_key_start_bit) +
                   " num_of_buckets_per_alloc=" + std::to_string(o.num_of_buckets_per_alloc) +
                   ">";
        });

    /* ─────────────────────  StringType Bindings  ───────────────────── */
    py::class_<mlkv::CuString>(m, "CuString", py::module_local(),
        "Simple fixed-size string that can be used as a value in the hash table (10 chars)")
        .def(py::init<>(), "Create an empty CuString")
        .def(py::init<const mlkv::CuString&>(), "Copy constructor")
        .def(py::init([](const char* str) {
            mlkv::CuString s;
            s = str;
            return s;
        }), py::arg("str"), "Create a CuString from C-style string")
        .def("__str__", [](const mlkv::CuString& s) { 
            return std::string(s.data); 
        })
        .def("__repr__", [](const mlkv::CuString& s) { 
            return "<CuString \"" + std::string(s.data) + "\">"; 
        })
        .def("__len__", [](const mlkv::CuString& s) {
            size_t len = 0;
            while (len < mlkv::MAX_STRING_LENGTH && s.data[len]) len++;
            return len;
        })
        .def("__getitem__", [](const mlkv::CuString& s, size_t i) {
            if (i >= mlkv::MAX_STRING_LENGTH) throw py::index_error("CuString index out of range");
            return s.data[i];
        })
        .def("__setitem__", [](mlkv::CuString& s, size_t i, char c) {
            if (i >= mlkv::MAX_STRING_LENGTH) throw py::index_error("CuString index out of range");
            s.data[i] = c;
        })
        .def("__eq__", [](const mlkv::CuString& s, const mlkv::CuString& other) { 
            return strcmp(s.data, other.data) == 0; 
        })
        .def("__eq__", [](const mlkv::CuString& s, const char* other) { 
            return strcmp(s.data, other ? other : "") == 0; 
        })
        .def("__iadd__", [](mlkv::CuString& s, const mlkv::CuString& other) { 
            s += other; 
            return s; 
        })
        .def("__iadd__", [](mlkv::CuString& s, const char* other) { 
            if (other) {
                size_t len = 0;
                while (s.data[len] && len < mlkv::MAX_STRING_LENGTH - 1) len++;
                
                size_t i = 0;
                while (other[i] && len + i < mlkv::MAX_STRING_LENGTH - 1) {
                    s.data[len + i] = other[i];
                    i++;
                }
                s.data[len + i] = '\0';
            }
            return s; 
        })
        .def("assign", [](mlkv::CuString& s, const char* str) {
            s = str;
            return s;
        }, py::arg("str"), "Assign a C-style string to this CuString")
        .def("c_str", [](const mlkv::CuString& s) { 
            return s.data; 
        }, "Get the C-style string data")
        .def("empty", [](const mlkv::CuString& s) { 
            return s.data[0] == '\0'; 
        }, "Check if the string is empty")
        .def("clear", [](mlkv::CuString& s) { 
            s.data[0] = '\0'; 
        }, "Clear the string")
        .def("max_size", [](const mlkv::CuString& s) { 
            return mlkv::MAX_STRING_LENGTH; 
        }, "Get the maximum size of the string buffer")
        .def_property_readonly_static("sizeof", [](py::object) { 
            return sizeof(mlkv::CuString); 
        }, "Get the size of this CuString object in bytes")
        .def_property_readonly_static("max_length", [](py::object) { 
            return mlkv::MAX_STRING_LENGTH; 
        }, "Get the maximum string length");

    // py::class_<mlkv::StringType<256>>(m, "String", py::module_local(), 
    //     "Fixed-size string that can be used as a value in the hash table (256 chars)")
    //     .def(py::init<>(), "Create an empty string")
    //     .def(py::init<const char*>(), "Create a string from C-style string")
    //     .def(py::init<const mlkv::StringType<256>&>(), "Copy constructor")
    //     .def("__str__", [](const mlkv::StringType<256>& s) { return std::string(s.c_str()); })
    //     .def("__repr__", [](const mlkv::StringType<256>& s) { 
    //         return "<String \"" + std::string(s.c_str()) + "\">"; 
    //     })
    //     .def("__len__", &mlkv::StringType<256>::length)
    //     .def("__getitem__", [](const mlkv::StringType<256>& s, size_t i) {
    //         if (i >= s.length()) throw py::index_error("String index out of range");
    //         return s[i];
    //     })
    //     .def("__eq__", [](const mlkv::StringType<256>& s, const char* other) { return s == other; })
    //     .def("__eq__", [](const mlkv::StringType<256>& s, const mlkv::StringType<256>& other) { return s == other; })
    //     .def("__add__", [](const mlkv::StringType<256>& s, const char* other) { return s + other; })
    //     .def("__add__", [](const mlkv::StringType<256>& s, const mlkv::StringType<256>& other) { return s + other; })
    //     .def("__iadd__", [](mlkv::StringType<256>& s, const char* other) { return s += other; })
    //     .def("__iadd__", [](mlkv::StringType<256>& s, const mlkv::StringType<256>& other) { return s += other; })
    //     .def("find", [](const mlkv::StringType<256>& s, const char* substr, size_t pos) { return s.find(substr, pos); },
    //         py::arg("substr"), py::arg("pos") = 0)
    //     .def("find", [](const mlkv::StringType<256>& s, char c, size_t pos) { return s.find(c, pos); },
    //         py::arg("char"), py::arg("pos") = 0)
    //     .def("substr", &mlkv::StringType<256>::substr,
    //         py::arg("pos") = 0, py::arg("len") = mlkv::StringType<256>::npos)
    //     .def("c_str", &mlkv::StringType<256>::c_str)
    //     .def("empty", &mlkv::StringType<256>::empty)
    //     .def("clear", &mlkv::StringType<256>::clear)
    //     .def("length", &mlkv::StringType<256>::length)
    //     .def("max_size", &mlkv::StringType<256>::max_size)
    //     .def("resize", &mlkv::StringType<256>::resize,
    //         py::arg("count"), py::arg("ch") = '\0')
    //     .def_property_readonly_static("sizeof", [](py::object) { return sizeof(mlkv::StringType<256>); },
    //         "Get the size of this StringType object in bytes.");


    // py::class_<mlkv::StringType<64>>(m, "SmallString", py::module_local(), 
    //     "Fixed-size string that can be used as a value in the hash table (64 chars)")
    //     .def(py::init<>(), "Create an empty string")
    //     .def(py::init<const char*>(), "Create a string from C-style string")
    //     .def(py::init<const mlkv::StringType<64>&>(), "Copy constructor")
    //     .def("__str__", [](const mlkv::StringType<64>& s) { return std::string(s.c_str()); })
    //     .def("__repr__", [](const mlkv::StringType<64>& s) { 
    //         return "<SmallString \"" + std::string(s.c_str()) + "\">"; 
    //     })
    //     .def("__len__", &mlkv::StringType<64>::length)
    //     .def("__getitem__", [](const mlkv::StringType<64>& s, size_t i) {
    //         if (i >= s.length()) throw py::index_error("String index out of range");
    //         return s[i];
    //     })
    //     .def("__eq__", [](const mlkv::StringType<64>& s, const char* other) { return s == other; })
    //     .def("__eq__", [](const mlkv::StringType<64>& s, const mlkv::StringType<64>& other) { return s == other; })
    //     .def("__add__", [](const mlkv::StringType<64>& s, const char* other) { return s + other; })
    //     .def("__add__", [](const mlkv::StringType<64>& s, const mlkv::StringType<64>& other) { return s + other; })
    //     .def("__iadd__", [](mlkv::StringType<64>& s, const char* other) { return s += other; })
    //     .def("__iadd__", [](mlkv::StringType<64>& s, const mlkv::StringType<64>& other) { return s += other; })
    //     .def("find", [](const mlkv::StringType<64>& s, const char* substr, size_t pos) { return s.find(substr, pos); },
    //         py::arg("substr"), py::arg("pos") = 0)
    //     .def("find", [](const mlkv::StringType<64>& s, char c, size_t pos) { return s.find(c, pos); },
    //         py::arg("char"), py::arg("pos") = 0)
    //     .def("substr", &mlkv::StringType<64>::substr,
    //         py::arg("pos") = 0, py::arg("len") = mlkv::StringType<64>::npos)
    //     .def("c_str", &mlkv::StringType<64>::c_str)
    //     .def("empty", &mlkv::StringType<64>::empty)
    //     .def("clear", &mlkv::StringType<64>::clear)
    //     .def("length", &mlkv::StringType<64>::length)
    //     .def("max_size", &mlkv::StringType<64>::max_size)
    //     .def("resize", &mlkv::StringType<64>::resize,
    //         py::arg("count"), py::arg("ch") = '\0')
    //     .def_property_readonly_static("sizeof", [](py::object) { return sizeof(mlkv::StringType<64>); },
    //         "Get the size of this StringType object in bytes.");


    // py::class_<mlkv::StringType<1024>>(m, "LargeString", py::module_local(), 
    //     "Fixed-size string that can be used as a value in the hash table (1024 chars)")
    //     .def(py::init<>(), "Create an empty string")
    //     .def(py::init<const char*>(), "Create a string from C-style string")
    //     .def(py::init<const mlkv::StringType<1024>&>(), "Copy constructor")
    //     .def("__str__", [](const mlkv::StringType<1024>& s) { return std::string(s.c_str()); })
    //     .def("__repr__", [](const mlkv::StringType<1024>& s) { 
    //         return "<LargeString \"" + std::string(s.c_str()) + "\">"; 
    //     })
    //     .def("__len__", &mlkv::StringType<1024>::length)
    //     .def("__getitem__", [](const mlkv::StringType<1024>& s, size_t i) {
    //         if (i >= s.length()) throw py::index_error("String index out of range");
    //         return s[i];
    //     })
    //     .def("__eq__", [](const mlkv::StringType<1024>& s, const char* other) { return s == other; })
    //     .def("__eq__", [](const mlkv::StringType<1024>& s, const mlkv::StringType<1024>& other) { return s == other; })
    //     .def("__add__", [](const mlkv::StringType<1024>& s, const char* other) { return s + other; })
    //     .def("__add__", [](const mlkv::StringType<1024>& s, const mlkv::StringType<1024>& other) { return s + other; })
    //     .def("__iadd__", [](mlkv::StringType<1024>& s, const char* other) { return s += other; })
    //     .def("__iadd__", [](mlkv::StringType<1024>& s, const mlkv::StringType<1024>& other) { return s += other; })
    //     .def("find", [](const mlkv::StringType<1024>& s, const char* substr, size_t pos) { return s.find(substr, pos); },
    //         py::arg("substr"), py::arg("pos") = 0)
    //     .def("find", [](const mlkv::StringType<1024>& s, char c, size_t pos) { return s.find(c, pos); },
    //         py::arg("char"), py::arg("pos") = 0)
    //     .def("substr", &mlkv::StringType<1024>::substr,
    //         py::arg("pos") = 0, py::arg("len") = mlkv::StringType<1024>::npos)
    //     .def("c_str", &mlkv::StringType<1024>::c_str)
    //     .def("empty", &mlkv::StringType<1024>::empty)
    //     .def("clear", &mlkv::StringType<1024>::clear)
    //     .def("length", &mlkv::StringType<1024>::length)
    //     .def("max_size", &mlkv::StringType<1024>::max_size)
    //     .def("resize", &mlkv::StringType<1024>::resize,
    //         py::arg("count"), py::arg("ch") = '\0')
    //     .def_property_readonly_static("sizeof", [](py::object) { return sizeof(mlkv::StringType<1024>); },
    //         "Get the size of this StringType object in bytes.");

    /* ─────────────────────  HashTable  ───────────────────── */
    // Define common types for HashTable
    using KeyType = uint64_t;       // K
    using ValueType = mlkv::CuString;        // V
    using ScoreType = uint64_t;     // S
    using ArchTag = Sm80;           // CUDA architecture tag
    
    // Default eviction strategy is LRU
    using HashTableType = nv::merlin::HashTable<KeyType, ValueType, ScoreType, EvictStrategy::kLru, ArchTag>;
    
    py::class_<HashTableType>(m, "HashTable", py::module_local(), "GPU-accelerated hash table with LRU eviction strategy.")
        .def(py::init<>(), "Default constructor for the hash table.")
        .def("init", 
             [](HashTableType& table, HashTableOptionsType options, py::object allocator) {
                 // Allocator should be nullptr by default
                 BaseAllocator* alloc_ptr = nullptr;
                 if (!allocator.is_none() && py::isinstance<BaseAllocator*>(allocator)) {
                     alloc_ptr = allocator.cast<BaseAllocator*>();
                 }
                 table.init(options, alloc_ptr);
             }, 
             py::arg("options"), 
             py::arg("allocator") = py::none(),
             "Initialize the hash table with the given options.")
        .def("insert_or_assign", 
             [](HashTableType& table,
                size_t n,
                std::uintptr_t keys_ptr,
                std::uintptr_t values_ptr,
                py::object scores_obj,
                py::object stream, 
                bool unique_key,
                bool ignore_evict_strategy) {
                 
                 // Handle scores (they can be None/nullptr)
                 ScoreType* scores_ptr = nullptr;
                 if (!scores_obj.is_none()) {
                     scores_ptr = reinterpret_cast<ScoreType*>(scores_obj.cast<std::uintptr_t>());
                 }

                 // Handle stream
                 cudaStream_t stream_ptr = 0;  // Default stream
                 if (!stream.is_none()) {
                     stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>());
                 }
                 
                 table.insert_or_assign(
                     n,
                     reinterpret_cast<KeyType*>(keys_ptr),
                     reinterpret_cast<ValueType*>(values_ptr),
                     scores_ptr,
                     stream_ptr,
                     unique_key,
                     ignore_evict_strategy
                 );
             },
             py::arg("n"),
             py::arg("keys_ptr"),
             py::arg("values_ptr"),
             py::arg("scores_ptr") = py::none(),
             py::arg("stream") = py::none(),
             py::arg("unique_key") = true,
             py::arg("ignore_evict_strategy") = false,
             "Insert new key-value-score tuples into the hash table, or update existing ones. "
             "For vector values, the values_ptr should point to a contiguous array of [n * dim] ValueType elements.")
        .def("find", 
             [](const HashTableType& table, 
                size_t n,
                std::uintptr_t keys_ptr,
                std::uintptr_t values_ptr,
                std::uintptr_t founds_ptr,
                py::object scores_obj,
                py::object stream) {
                 
                 // Handle scores (they can be None/nullptr)
                 ScoreType* scores_ptr = nullptr;
                 if (!scores_obj.is_none()) {
                     scores_ptr = reinterpret_cast<ScoreType*>(scores_obj.cast<std::uintptr_t>());
                 }
                 
                 // Handle stream
                 cudaStream_t stream_ptr = 0;  // Default stream
                 if (!stream.is_none()) {
                     stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>());
                 }
                 
                 table.find(
                     n,
                     reinterpret_cast<KeyType*>(keys_ptr),
                     reinterpret_cast<ValueType*>(values_ptr),
                     reinterpret_cast<bool*>(founds_ptr),
                     scores_ptr,
                     stream_ptr
                 );
             },
             py::arg("n"),
             py::arg("keys_ptr"),
             py::arg("values_ptr"),
             py::arg("founds_ptr"),
             py::arg("scores_ptr") = py::none(),
             py::arg("stream") = py::none(),
             "Search the hash table for the specified keys using raw pointers. Values are updated only when keys are found.")
        .def("size", 
             [](const HashTableType& table, py::object stream) {
                 cudaStream_t stream_ptr = 0;  // Default stream
                 if (!stream.is_none()) {
                     stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>());
                 }
                 return table.size(stream_ptr);
             },
             py::arg("stream") = py::none(),
             "Returns the number of elements in the hash table.")
        .def("capacity", &HashTableType::capacity,
             "Returns the capacity of the hash table.")
        .def("load_factor", 
             [](const HashTableType& table, py::object stream) {
                 cudaStream_t stream_ptr = 0;  // Default stream
                 if (!stream.is_none()) {
                     stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>());
                 }
                 return table.load_factor(stream_ptr);
             },
             py::arg("stream") = py::none(),
             "Returns the load factor of the hash table (size/capacity).")
        .def("set_global_epoch", &HashTableType::set_global_epoch, py::arg("epoch"),
             "Set the global epoch for eviction strategies that use epochs.")
        .def("dim", &HashTableType::dim,
             "Returns the dimension of the values stored in the hash table.");
}