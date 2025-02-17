/**
 * Test for verification of the generate sequential number
 * function gen_seq_num
 */

#include "gen_num.h"
#include <cassert>

int main() { // NOLINT
             // Define local array size, array pointer
  constexpr int test_array_size = 5;
  int start_number = 1;
  int my_array[test_array_size]; // NOLINT
  int *my_array_ptr = my_array;  // NOLINT

  // Fill array with sequential integers starting with
  // start_number
  gen_seq_num(my_array_ptr, test_array_size, start_number);

  // Calculate sum of array and verify with True sum
  int calc_sum = std::accumulate(my_array_ptr, my_array_ptr + test_array_size,
                                 0); // NOLINT
  constexpr int true_sum = 15;

  /*
  if (calc_sum == true_sum){
      return 0;
  } else { //NOLINT
      return 1;
  }
  */

  // Check for passing the test
  assert(calc_sum == true_sum); // NOLINT
}
