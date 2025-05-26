import merlin_hashtable_python.merlin_hashtable_python as mh
import cupy as cp
import numpy as np
import random
import string
import ctypes


device_id = 0

cp.cuda.Device(device_id).use()



def pack_strings(str_list, width=10):
    padded = [s.encode() if isinstance(s, str) else s for s in str_list]
    padded = [s.ljust(width, b'\0')[:width] for s in padded]
    return np.frombuffer(b''.join(padded), dtype=np.uint8).reshape(-1, width)

def unpack_strings(uint8_arr):
    return [bytes(row).rstrip(b'\0').decode() for row in uint8_arr]



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
    
    
    n = 3
    strs = ['hellohello', 'cuda', 'world']
    np_arr = pack_strings(strs)
    
    d_values = cp.asarray(np_arr)
    d_keys = cp.array([0, 1, 2], dtype=cp.uint64)
    d_values_out = cp.empty((n, options.dim*10), dtype=cp.uint8)
    d_found = cp.empty((n,), dtype=cp.bool_)
    stream = cp.cuda.Stream()
    
    table.insert_or_assign(
        n,
        d_keys.data.ptr,
        d_values.data.ptr,
        stream=stream.ptr
    )
    stream.synchronize()
    
    
    table.find(n, d_keys.data.ptr, d_values_out.data.ptr, d_found.data.ptr, stream=stream.ptr)
    stream.synchronize()
    
    print(unpack_strings(d_values_out))
    
    
    
    

if __name__ == "__main__":
    # # Run the original byte array test
    # test_with_bytes()
    
    test_with_stringtype()
    