cmake_minimum_required(VERSION 3.9.1)
project(DotProduct)
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include(CTest)

# Set properties
set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_CLANG_TIDY clang-tidy; -header-filter=.; -checks=*; --format-style='google';)
set(SOURCES gen_num.cpp main.cpp)

# Create executable
add_library(gen_num gen_num.cpp)
add_executable(dot_product ${SOURCES})
target_link_libraries(dot_product ${MPI_CXX_LIBRARIES})

# Testing binary
add_executable(gen_num_test test_gen_num.cpp)
target_link_libraries(gen_num_test gen_num)

# Compile with coverage options
target_compile_options(gen_num_test PRIVATE -coverage)
target_link_options(gen_num_test PRIVATE -coverage)

# Testing binary
add_executable(dot_product_test test_dot_product.cpp)
target_link_libraries(dot_product_test gen_num ${MPI_CXX_LIBRARIES})

# Compile with coverage options
target_compile_options(dot_product_test PRIVATE -coverage)
target_link_options(dot_product_test PRIVATE -coverage)

# Enable testing functionality
enable_testing()

# Define tests
add_test(NAME gen_num_test COMMAND $<TARGET_FILE:gen_num_test>)
add_test(NAME dot_product_test COMMAND $<TARGET_FILE:dot_product_test>)
