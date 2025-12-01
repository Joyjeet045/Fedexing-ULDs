cmake_minimum_required(VERSION 3.20.0)
project(fedex)

add_executable(main main.cpp bin_pack.h packing_structures.h)
target_compile_features(main PRIVATE cxx_std_20)