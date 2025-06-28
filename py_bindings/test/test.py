import merlin_hashtable_python.merlin_hashtable_python as mh
import cupy as cp
import numpy as np
import random
import string
import ctypes


device_id = 0

cp.cuda.Device(device_id).use()



def pack_strings(str_list, width=10):
    """
    Pack a 2D array of strings into a uint8 numpy array
    
    Args:
        str_list: 2D list of strings, shape (n_rows, n_cols)
        width: maximum width for each string (will be padded/truncated to this size)
    
    Returns:
        numpy array of shape (n_rows, n_cols * width) with dtype uint8
    """
    if not str_list:
        return np.array([], dtype=np.uint8)
    
    # Handle both 1D and 2D cases
    if isinstance(str_list[0], str):
        # 1D case - convert to 2D with single row
        str_list = [str_list]
    
    n_rows = len(str_list)
    n_cols = len(str_list[0]) if str_list else 0
    
    # Create output array
    result = np.zeros((n_rows, n_cols * width), dtype=np.uint8)
    
    for i, row in enumerate(str_list):
        for j, s in enumerate(row):
            # Convert string to bytes if needed
            if isinstance(s, str):
                s_bytes = s.encode('utf-8')
            else:
                s_bytes = s
            
            # Pad or truncate to width
            s_padded = s_bytes.ljust(width, b'\0')[:width]
            
            # Copy to result array
            start_idx = j * width
            end_idx = start_idx + width
            result[i, start_idx:end_idx] = np.frombuffer(s_padded, dtype=np.uint8)
    
    return result

def unpack_strings(uint8_arr, width=10):
    """
    Unpack a uint8 numpy array back to 2D list of strings
    
    Args:
        uint8_arr: numpy array of shape (n_rows, n_cols * width) with dtype uint8
        width: width of each packed string
    
    Returns:
        2D list of strings
    """
    if uint8_arr.size == 0:
        return []
    
    n_rows, total_width = uint8_arr.shape
    n_cols = total_width // width
    
    result = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            start_idx = j * width
            end_idx = start_idx + width
            string_bytes = uint8_arr[i, start_idx:end_idx].tobytes()
            # Remove null padding and decode
            string_val = string_bytes.rstrip(b'\0').decode('utf-8')
            row.append(string_val)
        result.append(row)
    
    return result



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
    options.dim = 5  # Each key corresponds to one string
    options.init_capacity = 1024 * 1024  # 1M
    options.max_capacity = 1024 * 1024   # 1M
    options.max_hbm_for_vectors = 4 * 1024 * 1024 * 1024  # 4GB
    options.io_by_cpu = False
    options.device_id = device_id

    table = mh.HashTable()
    table.init(options)

    print("Hash table initialization complete")
    
    
    n = 3
    strs = [
        ['hellohello', 'cudacuda', 'worldworld', 'hellohello', 'cudacuda'],
        ['hellohello', 'cudacuda', 'worldworld', 'hellohello', 'cudacuda'],  
        ['hellohello', 'cudacuda', 'worldworld', 'hellohello', 'cudacuda']
    ]
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
    
    print(unpack_strings(d_values_out, width=10))
    
    
    
    

if __name__ == "__main__":
    # # Run the original byte array test
    # test_with_bytes()
    
    test_with_stringtype()
    