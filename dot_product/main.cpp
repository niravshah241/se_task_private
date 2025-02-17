/**
 * Parallel dot product example
 */
#include "gen_num.h"

int main(int argc, char *argv[]) { // NOLINT
  // Initalisation of MPI
  int size; // NOLINT
  int rank; // NOLINT
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Define local array size, pointer arrays
  constexpr int local_array_size = 10;
  int array_start_num = rank * local_array_size + 1;
  int rand_num_array[local_array_size]; // NOLINT
  int *array_ptr = rand_num_array;      // NOLINT

  // Fill array with sequential integers starting with
  // array_start_num
  gen_seq_num(array_ptr, local_array_size, array_start_num);

  // Perform dot product over local array
  int dot_product = 0;
  dot_product = calc_dot_product(array_ptr, local_array_size);

  // Perform dot product over array from all processes (global)
  int global_dot_product = 0;
  MPI_Allreduce(&dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  std::cout << "Calculated dot product: " << global_dot_product << std::endl;

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
