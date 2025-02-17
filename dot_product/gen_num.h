#ifndef GENNUM_H // NOLINT
#define GENNUM_H // NOLINT

#include "mpi.h"
#include <algorithm>
#include <iostream>
#include <numeric>

auto gen_seq_num(int *rand_num_array, int array_size, // NOLINT
                 int start_num) -> void;              // NOLINT

auto calc_dot_product(int *rand_num_array,          // NOLINT
                      int local_array_size) -> int; // NOLINT

#endif
