cmake_minimum_required(VERSION 3.6)
project(sturdr)

string(ASCII 27 Esc)
message(STATUS "${Esc}[1;31mBuild type: ${CMAKE_BUILD_TYPE}${Esc}[m") 

# --- Dependencies ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)
add_subdirectory(navtools)
add_subdirectory(satutils)
add_subdirectory(sturdds)
add_subdirectory(sturdio)
add_subdirectory(sturdins)
add_subdirectory(sturdr)

# --- Executables ---
# add_executable(test_gps_ins_sim tests/test_gps_ins_sim.cpp)
# target_link_libraries(test_gps_ins_sim 
#   PUBLIC 
#     Eigen3::Eigen
#     navtools
#     gps-ins
# )