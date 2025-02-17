#include "gen_num.h"

auto gen_seq_num(int *rand_num_array, int array_size, // NOLINT
                 int start_num) -> void {             // NOLINT
  /**
   * Function to generate array of sequential integer numbers.
   * The provided array is filled with sequential integer
   * numbers.
   * Arguments:
   *     rand_num_array (int*): Pointer to integer array
   *     array_size (int): Size of the array
   *     start_num (int): Starting number of array
   */

  /*
  for (int i = 0; i < array_size; i++) {
    *(rand_num_array + i) = i + start_num; //NOLINT
  };
  */

  std::iota(rand_num_array, rand_num_array + array_size, start_num); // NOLINT

  /*
  for (int i = 0; i < array_size; i++) {
    std::cout << rand_num_array[i] << std::endl;
  };
  */
};

auto calc_dot_product(int *rand_num_array,           // NOLINT
                      int local_array_size) -> int { // NOLINT
  /**
   * Function to calculate dot product
   * Arguments:
   *     rand_num_array (int*): Pointer to integer array
   *     local_array_size (int): Local size of the array
   */
  int product = std::inner_product(rand_num_array,
                                   rand_num_array + local_array_size, // NOLINT
                                   rand_num_array, 0);                // NOLINT
  return product;
};
