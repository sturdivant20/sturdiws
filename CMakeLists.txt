cmake_minimum_required(VERSION 3.6)
project(sturdr)

string(ASCII 27 Esc)
message(STATUS "${Esc}[1;31mBuild type: ${CMAKE_BUILD_TYPE}${Esc}[m") 

# --- Dependencies ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/src" ${CMAKE_PREFIX_PATH})

# --- Compiler Flags ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -W")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -fomit-frame-pointer -fstrict-aliasing -flto -DNDEBUG -march=native")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(src/navtools)
add_subdirectory(src/satutils)
# add_subdirectory(src/sturdds)
add_subdirectory(src/sturdio)
add_subdirectory(src/sturdins)
add_subdirectory(src/sturdr)

# --- Executables ---
# add_executable(test_gps_ins_sim tests/test_gps_ins_sim.cpp)
# target_link_libraries(test_gps_ins_sim 
#   PUBLIC 
#     Eigen3::Eigen
#     navtools
#     gps-ins
# )