/**
 * Test for verification of the dot product
 * function calc_dot_product
 */

#include "gen_num.h"
#include <cassert>

int main(int argc, char *argv[]) { // NOLINT
  // Initalization of MPI
  int size; // NOLINT
  int rank; // NOLINT
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Define local array size, array pointer
  constexpr int test_array_size = 5;
  int start_number = rank * test_array_size + 1;
  int my_array[test_array_size]; // NOLINT
  int *my_array_ptr = my_array;  // NOLINT
  // Fill array with sequential integers starting with
  // start_number
  gen_seq_num(my_array_ptr, test_array_size, start_number);

  // Perform dot product over local array
  int test_dot_product = calc_dot_product(my_array_ptr, test_array_size);

  // Perform dot product over array from all processes (global)
  int test_global_dot_product = 0;
  MPI_Allreduce(&test_dot_product, &test_global_dot_product, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  // Verification with analytical result
  int num_terms = test_array_size * size;
  int true_global_dot_product =
      num_terms * (num_terms + 1) * (2 * num_terms + 1) / 6; // NOLINT

  // Check for passing the test
  assert(test_global_dot_product == true_global_dot_product); // NOLINT

  MPI_Finalize();

  /*
  if (test_global_dot_product == true_global_dot_product){
      return 0;
  } else { //NOLINT
      return 1;
  }
  */
}
