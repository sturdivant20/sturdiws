cmake_minimum_required(VERSION 3.6)
project(sturdiws)

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
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -fomit-frame-pointer -fstrict-aliasing -flto -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -DNDEBUG -march=native -fPIC")

add_subdirectory(src/navtools)
add_subdirectory(src/satutils)
add_subdirectory(src/navsim)
add_subdirectory(src/sturdio)
add_subdirectory(src/sturdins)
add_subdirectory(src/sturdr)

# # --- SturDR (SDR) ---
# if (NOT DEFINED INSTALL_STURDR OR INSTALL_STURDR)
#     add_subdirectory(src/sturdr)
#     add_executable(run_sturdr scripts/run_sturdr.cpp)
#     target_link_libraries(run_sturdr PUBLIC 
#         navtools
#         satutils
#         sturdio
#         sturdins
#         sturdr
#     )
# endif()

# # --- Navsim (Simulations) ---
# if (NOT DEFINED INSTALL_SIMULATOR OR INSTALL_SIMULATOR)
#     add_subdirectory(src/navsim)
#     add_executable(run_correlator_sim scripts/run_correlator_sim.cpp)
#     target_link_libraries(run_correlator_sim PUBLIC 
#         navtools
#         satutils
#         navsim
#         sturdio
#         sturdins
#     )
# endif()