import merlin_hashtable_python.merlin_hashtable_python as mh
import cupy as cp
import numpy as np
import random
import string
import ctypes


device_id = 0

cp.cuda.Device(device_id).use()


def test_with_bytes():
    """Original test using byte arrays as values"""
    print("\n===== Testing Byte Array Values =====")
    
    options = mh.HashTableOptions()
    options.dim = 5  
    options.init_capacity = 64 * 1024 * 1024
    options.max_capacity = 64 * 1024 * 1024
    options.max_hbm_for_vectors = 18 * 1024 * 1024 * 1024
    options.io_by_cpu = False
    options.device_id = device_id


    table = mh.HashTable()
    table.init(options)


    n = 100  # Number of keys to handle
    # d_keys = cp.array([1, 2, 3], dtype=cp.uint64)


    d_values = cp.random.randint(32, 127, size=(n, options.dim), dtype=cp.int8)
    d_values_out = cp.empty((n, options.dim), dtype=cp.int8)
    d_found = cp.empty((n,), dtype=cp.bool_)
    stream = cp.cuda.Stream()

    # Create a random array of keys
    d_keys = cp.arange(n, dtype=cp.uint64)

    # Insert the keys and values into the hash table
    table.insert_or_assign(
        n,
        d_keys.data.ptr,
        d_values.data.ptr,
        stream=stream.ptr
    )
    stream.synchronize()



    table.find(n, d_keys.data.ptr, d_values_out.data.ptr, d_found.data.ptr, stream=stream.ptr)

    stream.synchronize()


    print("Keys:")
    print(d_keys)
    print("Values inserted (as bytes):")
    print(d_values)
    print("Values inserted (as characters):")

    for i in range(n):
        chars = ''.join([chr(c) for c in d_values[i].get()])
        print(f"Key {d_keys[i].get()}: '{chars}'")

    print("Values retrieved (as bytes):")
    print(d_values_out)
    print("Values retrieved (as characters):")
    for i in range(n):
        chars = ''.join([chr(c) for c in d_values_out[i].get()])
        print(f"Key {d_keys[i].get()}: '{chars}'")

    print("Found status:")
    print(d_found)


    if cp.all(d_found):
        print("All keys were found!")
        equal = cp.all(d_values == d_values_out)
        
        print(f"Values match: {equal}")
        if equal:
            print("Values match correctly!")
    else:
        print(f"Only {cp.sum(d_found)}/{n} keys were found.")


def random_string(length):
    """Generate a random string of specified length"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def test_with_stringtype():
    """Test using StringType objects as values"""
    print("\n===== Testing StringType Values =====")
    
    # Create and initialize hash table
    options = mh.HashTableOptions()
    options.dim = 1  # Each key corresponds to one string
    options.init_capacity = 1024 * 1024  # 1M
    options.max_capacity = 1024 * 1024   # 1M
    options.max_hbm_for_vectors = 4 * 1024 * 1024 * 1024  # 4GB
    options.io_by_cpu = False
    options.device_id = device_id

    table = mh.HashTable()
    table.init(options)

    print("Hash table initialization complete")

    # Test parameters
    n = 100  # Number of key-value pairs to test
    string_length = 20  # String length

    print(f"Preparing to test storing and retrieving {n} string values...")

    # Use CuPy to allocate device memory
    d_keys = cp.arange(n, dtype=cp.uint64)

    # Create StringType objects for testing
    # 1. First allocate StringType objects on host
    host_strings = []
    for i in range(n):
        # Create a SmallString object
        str_obj = mh.SmallString(random_string(string_length))
        host_strings.append(str_obj)
        
    # 2. Allocate device memory
    # Use the built-in sizeof attribute instead of ctypes.sizeof
    sizeof_string = mh.SmallString.sizeof
    d_values = cp.cuda.memory.alloc(n * sizeof_string)
    d_values_out = cp.cuda.memory.alloc(n * sizeof_string)
    d_found = cp.empty((n,), dtype=cp.bool_)

    # Create CUDA stream
    stream = cp.cuda.Stream()

    # Create a numpy array on host to hold string objects
    host_values_mem = np.zeros((n * sizeof_string,), dtype=np.uint8)
    
    # 3. Copy StringType objects from host to device
    for i in range(n):
        # Get string object and its content
        str_obj = host_strings[i]
        str_content = str(str_obj)
        
        # Manually copy the bytes from our string to host_values_mem
        host_ptr = host_values_mem[i * sizeof_string:(i + 1) * sizeof_string]
        
        # Convert the string content to bytes and copy to buffer
        # First, create a C buffer for the string content
        str_bytes = str_content.encode('utf-8')
        # Copy the string bytes to our buffer
        for j in range(min(len(str_bytes), sizeof_string - 1)):
            host_ptr[j] = str_bytes[j]
        # Ensure null termination
        if len(str_bytes) < sizeof_string:
            host_ptr[len(str_bytes)] = 0
    
    # Now copy the entire numpy array to device at once
    cp.cuda.runtime.memcpy(
        d_values.ptr,
        host_values_mem.ctypes.data,
        n * sizeof_string,
        cp.cuda.runtime.memcpyHostToDevice
    )

    # 4. Insert key-value pairs into hash table
    table.insert_or_assign(
        n,
        d_keys.data.ptr,
        d_values.ptr,
        stream=stream.ptr
    )
    stream.synchronize()

    print("Key-value pairs inserted into hash table")

    # 5. Retrieve values from hash table
    table.find(
        n, 
        d_keys.data.ptr, 
        d_values_out.ptr, 
        d_found.data.ptr, 
        stream=stream.ptr
    )
    stream.synchronize()

    print("Values retrieved from hash table")

    # 6. Copy retrieved values from device back to host
    host_values_out_mem = np.zeros((n * sizeof_string,), dtype=np.uint8)
    
    # Copy all values back in one operation
    cp.cuda.runtime.memcpy(
        host_values_out_mem.ctypes.data,
        d_values_out.ptr,
        n * sizeof_string,
        cp.cuda.runtime.memcpyDeviceToHost
    )
    
    # Create StringType objects from the raw memory
    host_strings_out = []
    for i in range(n):
        # Get the memory chunk for this string
        str_data = host_values_out_mem[i * sizeof_string:(i + 1) * sizeof_string]
        
        # Convert bytes to string - find null terminator
        str_len = 0
        while str_len < sizeof_string and str_data[str_len] != 0:
            str_len += 1
            
        # Extract the string content
        c_str = bytes(str_data[:str_len]).decode('utf-8', errors='replace')
        
        # Create a fresh SmallString with the extracted content
        str_obj = mh.SmallString(c_str)
        host_strings_out.append(str_obj)

    # 7. Verify results
    print("\nVerification results:")
    print("Keys:")
    print(d_keys[:5])  # Only show first 5

    print("\nOriginal string values (first 5):")
    for i in range(min(5, n)):
        print(f"Key {d_keys[i].get()}: '{host_strings[i]}'")

    print("\nRetrieved string values (first 5):")
    for i in range(min(5, n)):
        print(f"Key {d_keys[i].get()}: '{host_strings_out[i]}'")

    print("\nSearch status:")
    found_array = d_found.get()
    print(f"Found: {np.sum(found_array)}/{n}")

    # 8. Verify all strings match
    matches = 0
    for i in range(n):
        if found_array[i] and str(host_strings[i]) == str(host_strings_out[i]):
            matches += 1

    print(f"\nString matches: {matches}/{n}")
    if matches == n:
        print("All strings match correctly! Test passed!")
    else:
        print("Some strings do not match. Test failed.")


    # Memory is automatically released when the pointers are garbage collected
    cp.get_default_memory_pool().free_all_blocks()

    print("StringType test complete")


if __name__ == "__main__":
    # Run the original byte array test
    test_with_bytes()
    
    # Run the StringType test
    test_with_stringtype()