cmake_minimum_required(VERSION 3.11)
project(Mandelbrot LANGUAGES CUDA CXX )

set(CMAKE_CXX_STANDARD 17 )

# To use SSE in fpng
# Add compiler flags: -msse4.1 -mpclmul -msse2")
# and set FPNG_NO_SSE=0
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wconversion")

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.h)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

add_executable(Mandelbrot ${CPP_SOURCES} ${CUDA_SOURCES})
target_include_directories(Mandelbrot PRIVATE include)
target_compile_definitions(Mandelbrot PUBLIC WITH_CUDA FPNG_NO_SSE=1)
target_link_libraries(Mandelbrot pthread png)

add_executable(Mandelbrot_cpu ${CPP_SOURCES})
target_include_directories(Mandelbrot_cpu PRIVATE include)
target_compile_definitions(Mandelbrot_cpu PUBLIC FPNG_NO_SSE=1)
target_link_libraries(Mandelbrot_cpu pthread png)
