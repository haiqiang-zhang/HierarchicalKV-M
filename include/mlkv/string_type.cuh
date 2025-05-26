#pragma once

#include <string>
#include <stdexcept>
#include <algorithm>

namespace mlkv {

// Define the fixed string length
const size_t MAX_STRING_LENGTH = 10; // Adjust this size as needed

// Custom string type that is compatible with CUDA and Merlin library
struct CuString {
  char data[MAX_STRING_LENGTH];
  
  __host__ __device__ CuString() {
    memset(data, 0, MAX_STRING_LENGTH);
  }
  
  __host__ __device__ CuString(const CuString& other) {
    memcpy(data, other.data, MAX_STRING_LENGTH);
  }
  
  __host__ __device__ CuString& operator=(const CuString& other) {
    if (this != &other) {
      memcpy(data, other.data, MAX_STRING_LENGTH);
    }
    return *this;
  }
  
  __host__ __device__ CuString& operator=(const char* str) {
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
  
  __host__ __device__ CuString& operator+=(const CuString& other) {
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

// namespace detail {
// /**
//  * @brief CUDA-compatible implementation of strcmp
//  */
// __host__ __device__ inline int device_strcmp(const char* s1, const char* s2) {
//   while (*s1 && (*s1 == *s2)) {
//     s1++;
//     s2++;
//   }
//   return *(const unsigned char*)s1 - *(const unsigned char*)s2;
// }

// /**
//  * @brief CUDA-compatible implementation of strstr
//  */
// __host__ __device__ inline const char* device_strstr(const char* haystack, const char* needle) {
//   if (!*needle) return haystack;
  
//   const char* p1;
//   const char* p2;
//   const char* p1_adv = haystack;
  
//   while (*p1_adv) {
//     p1 = p1_adv;
//     p2 = needle;
    
//     while (*p1 && *p2 && (*p1 == *p2)) {
//       p1++;
//       p2++;
//     }
    
//     if (!*p2) return p1_adv;
//     p1_adv++;
//   }
  
//   return nullptr;
// }
// } // namespace detail

// /**
//  * @brief A string class that can be used as value (V) in HierarchicalKV hashtable.
//  * 
//  * This class is CUDA-compatible and provides a fixed-size string buffer with
//  * basic string manipulation functionality. It is designed to be used with both 
//  * host and device code.
//  * 
//  * @tparam MaxLen The maximum length of the string (including null terminator).
//  */
// template <size_t MaxLen = 256>
// class StringType {
//  public:
//   using size_type = size_t;
//   using value_type = char;
//   using pointer = char*;
//   using const_pointer = const char*;
//   using reference = char&;
//   using const_reference = const char&;

//  private:
//   char data_[MaxLen];

//  public:
//   /**
//    * @brief Default constructor, creates an empty string.
//    */
//   __host__ __device__ StringType() {
//     memset(data_, 0, MaxLen);
//   }

//   /**
//    * @brief Constructor from C-style string.
//    * 
//    * @param str The C-style string to copy.
//    */
//   __host__ __device__ StringType(const char* str) {
//     assign(str);
//   }

//   /**
//    * @brief Copy constructor.
//    * 
//    * @param other The StringType to copy.
//    */
//   __host__ __device__ StringType(const StringType& other) {
//     memcpy(data_, other.data_, MaxLen);
//   }

//   /**
//    * @brief Assignment operator from another StringType.
//    * 
//    * @param other The StringType to copy.
//    * @return Reference to this.
//    */
//   __host__ __device__ StringType& operator=(const StringType& other) {
//     if (this != &other) {
//       memcpy(data_, other.data_, MaxLen);
//     }
//     return *this;
//   }

//   /**
//    * @brief Assignment operator from C-style string.
//    * 
//    * @param str The C-style string to copy.
//    * @return Reference to this.
//    */
//   __host__ __device__ StringType& operator=(const char* str) {
//     assign(str);
//     return *this;
//   }

//   /**
//    * @brief Assign a C-style string to this StringType.
//    * 
//    * @param str The C-style string to copy.
//    * @return Reference to this.
//    */
//   __host__ __device__ StringType& assign(const char* str) {
//     if (str) {
//       size_type len = 0;
//       while (str[len] && len < MaxLen - 1) len++;
//       memcpy(data_, str, len);
//       data_[len] = '\0';
//     } else {
//       data_[0] = '\0';
//     }
//     return *this;
//   }

//   /**
//    * @brief Append another string to this StringType.
//    * 
//    * @param other The StringType to append.
//    * @return Reference to this.
//    */
//   __host__ __device__ StringType& operator+=(const StringType& other) {
//     size_type len = length();
//     size_type i = 0;
//     const char* other_data = other.data_;
    
//     while (other_data[i] && len + i < MaxLen - 1) {
//       data_[len + i] = other_data[i];
//       i++;
//     }
//     data_[len + i] = '\0';
//     return *this;
//   }

//   /**
//    * @brief Append a C-style string to this StringType.
//    * 
//    * @param str The C-style string to append.
//    * @return Reference to this.
//    */
//   __host__ __device__ StringType& operator+=(const char* str) {
//     if (!str) return *this;
    
//     size_type len = length();
//     size_type i = 0;
    
//     while (str[i] && len + i < MaxLen - 1) {
//       data_[len + i] = str[i];
//       i++;
//     }
//     data_[len + i] = '\0';
//     return *this;
//   }

//   /**
//    * @brief Concatenate two StringType objects.
//    * 
//    * @param other The StringType to concatenate with.
//    * @return A new StringType with the concatenated contents.
//    */
//   __host__ __device__ StringType operator+(const StringType& other) const {
//     StringType result(*this);
//     result += other;
//     return result;
//   }

//   /**
//    * @brief Concatenate with a C-style string.
//    * 
//    * @param str The C-style string to concatenate with.
//    * @return A new StringType with the concatenated contents.
//    */
//   __host__ __device__ StringType operator+(const char* str) const {
//     StringType result(*this);
//     result += str;
//     return result;
//   }

//   /**
//    * @brief Access character at specified position.
//    * 
//    * @param pos The position to access.
//    * @return Reference to the character at position.
//    */
//   __host__ __device__ reference operator[](size_type pos) {
//     return data_[pos];
//   }

//   /**
//    * @brief Access character at specified position (const version).
//    * 
//    * @param pos The position to access.
//    * @return Const reference to the character at position.
//    */
//   __host__ __device__ const_reference operator[](size_type pos) const {
//     return data_[pos];
//   }

//   /**
//    * @brief Get the C-style string data.
//    * 
//    * @return Pointer to the C-style string data.
//    */
//   __host__ __device__ const_pointer c_str() const {
//     return data_;
//   }

//   /**
//    * @brief Get the string data.
//    * 
//    * @return Pointer to the string data.
//    */
//   __host__ __device__ pointer data() {
//     return data_;
//   }

//   /**
//    * @brief Get the string data (const version).
//    * 
//    * @return Const pointer to the string data.
//    */
//   __host__ __device__ const_pointer data() const {
//     return data_;
//   }

//   /**
//    * @brief Get the length of the string.
//    * 
//    * @return Length of the string (not including null terminator).
//    */
//   __host__ __device__ size_type length() const {
//     size_type len = 0;
//     while (len < MaxLen && data_[len]) len++;
//     return len;
//   }

//   /**
//    * @brief Get the maximum size of the string buffer.
//    * 
//    * @return Maximum size of the string buffer.
//    */
//   __host__ __device__ constexpr size_type max_size() const {
//     return MaxLen;
//   }

//   /**
//    * @brief Check if the string is empty.
//    * 
//    * @return True if the string is empty, false otherwise.
//    */
//   __host__ __device__ bool empty() const {
//     return data_[0] == '\0';
//   }

//   /**
//    * @brief Clear the string.
//    */
//   __host__ __device__ void clear() {
//     data_[0] = '\0';
//   }

//   /**
//    * @brief Resize the string.
//    * 
//    * @param count New size of the string.
//    * @param ch Character to fill with if expanding.
//    */
//   __host__ __device__ void resize(size_type count, char ch = '\0') {
//     size_type len = length();
    
//     if (count < len) {
//       // Truncate
//       data_[count] = '\0';
//     } else if (count < MaxLen) {
//       // Expand and fill with ch
//       for (size_type i = len; i < count; i++) {
//         data_[i] = ch;
//       }
//       data_[count] = '\0';
//     }
//   }

//   /**
//    * @brief Compare this StringType with another.
//    * 
//    * @param other The StringType to compare with.
//    * @return 0 if equal, negative if this is less, positive if this is greater.
//    */
//   __host__ __device__ int compare(const StringType& other) const {
//     return detail::device_strcmp(data_, other.data_);
//   }

//   /**
//    * @brief Compare this StringType with a C-style string.
//    * 
//    * @param str The C-style string to compare with.
//    * @return 0 if equal, negative if this is less, positive if this is greater.
//    */
//   __host__ __device__ int compare(const char* str) const {
//     return detail::device_strcmp(data_, str ? str : "");
//   }

//   /**
//    * @brief Find a substring in this StringType.
//    * 
//    * @param str The substring to find.
//    * @param pos The position to start searching from.
//    * @return Position of the found substring, or npos if not found.
//    */
//   __host__ __device__ size_type find(const char* str, size_type pos = 0) const {
//     if (!str || pos >= length()) return npos;
    
//     const char* result = detail::device_strstr(data_ + pos, str);
//     return result ? result - data_ : npos;
//   }

//   /**
//    * @brief Find a character in this StringType.
//    * 
//    * @param ch The character to find.
//    * @param pos The position to start searching from.
//    * @return Position of the found character, or npos if not found.
//    */
//   __host__ __device__ size_type find(char ch, size_type pos = 0) const {
//     size_type len = length();
//     for (size_type i = pos; i < len; i++) {
//       if (data_[i] == ch) return i;
//     }
//     return npos;
//   }

//   /**
//    * @brief Extract a substring from this StringType.
//    * 
//    * @param pos The starting position.
//    * @param len The length of the substring.
//    * @return A new StringType containing the substring.
//    */
//   __host__ __device__ StringType substr(size_type pos = 0, size_type len = npos) const {
//     StringType result;
//     size_type str_len = length();
    
//     if (pos > str_len) return result;
    
//     if (len == npos || pos + len > str_len) {
//       len = str_len - pos;
//     }
    
//     for (size_type i = 0; i < len; i++) {
//       result.data_[i] = data_[pos + i];
//     }
//     result.data_[len] = '\0';
    
//     return result;
//   }

//   /**
//    * @brief Equality operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if strings are equal, false otherwise.
//    */
//   __host__ __device__ bool operator==(const StringType& other) const {
//     return compare(other) == 0;
//   }

//   /**
//    * @brief Equality operator with C-style string.
//    * 
//    * @param str The C-style string to compare with.
//    * @return True if strings are equal, false otherwise.
//    */
//   __host__ __device__ bool operator==(const char* str) const {
//     return compare(str) == 0;
//   }

//   /**
//    * @brief Inequality operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if strings are not equal, false otherwise.
//    */
//   __host__ __device__ bool operator!=(const StringType& other) const {
//     return compare(other) != 0;
//   }

//   /**
//    * @brief Inequality operator with C-style string.
//    * 
//    * @param str The C-style string to compare with.
//    * @return True if strings are not equal, false otherwise.
//    */
//   __host__ __device__ bool operator!=(const char* str) const {
//     return compare(str) != 0;
//   }

//   /**
//    * @brief Less-than operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if this is less than other, false otherwise.
//    */
//   __host__ __device__ bool operator<(const StringType& other) const {
//     return compare(other) < 0;
//   }

//   /**
//    * @brief Greater-than operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if this is greater than other, false otherwise.
//    */
//   __host__ __device__ bool operator>(const StringType& other) const {
//     return compare(other) > 0;
//   }

//   /**
//    * @brief Less-than-or-equal operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if this is less than or equal to other, false otherwise.
//    */
//   __host__ __device__ bool operator<=(const StringType& other) const {
//     return compare(other) <= 0;
//   }

//   /**
//    * @brief Greater-than-or-equal operator.
//    * 
//    * @param other The StringType to compare with.
//    * @return True if this is greater than or equal to other, false otherwise.
//    */
//   __host__ __device__ bool operator>=(const StringType& other) const {
//     return compare(other) >= 0;
//   }

//   /**
//    * @brief Static constant representing "not found" value.
//    */
//   static constexpr size_type npos = static_cast<size_type>(-1);
// };

// /**
//  * @brief Concatenate a C-style string with a StringType.
//  * 
//  * @param lhs The C-style string.
//  * @param rhs The StringType.
//  * @return A new StringType with the concatenated contents.
//  */
// template <size_t MaxLen>
// __host__ __device__ StringType<MaxLen> operator+(const char* lhs, const StringType<MaxLen>& rhs) {
//   StringType<MaxLen> result(lhs);
//   result += rhs;
//   return result;
// }

// /**
//  * @brief Equality operator between C-style string and StringType.
//  * 
//  * @param lhs The C-style string.
//  * @param rhs The StringType.
//  * @return True if strings are equal, false otherwise.
//  */
// template <size_t MaxLen>
// __host__ __device__ bool operator==(const char* lhs, const StringType<MaxLen>& rhs) {
//   return rhs == lhs;
// }

// /**
//  * @brief Inequality operator between C-style string and StringType.
//  * 
//  * @param lhs The C-style string.
//  * @param rhs The StringType.
//  * @return True if strings are not equal, false otherwise.
//  */
// template <size_t MaxLen>
// __host__ __device__ bool operator!=(const char* lhs, const StringType<MaxLen>& rhs) {
//   return rhs != lhs;
// }

// /**
//  * @brief A convenience type alias for a default StringType with 256 characters.
//  */
// using String = StringType<256>;

// /**
//  * @brief A convenience type alias for a small StringType with 64 characters.
//  */
// using SmallString = StringType<64>;

// /**
//  * @brief A convenience type alias for a large StringType with 1024 characters.
//  */
// using LargeString = StringType<1024>;

}