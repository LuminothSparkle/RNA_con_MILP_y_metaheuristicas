cmake_minimum_required(VERSION 3.20)

project("GRB")

option(CXX "enable C++ compilation" ON)

if(CXX)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 23)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Visual Studio compiler with static runtime libraries
if(MSVC AND MT)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()

find_package(GUROBI REQUIRED)

include_directories(${GUROBI_INCLUDE_DIRS})

file(GLOB SOURCES
  "*.h"
  "*.cpp"
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

if(CXX)
  target_link_libraries(${CMAKE_PROJECT_NAME}
    optimized ${GUROBI_CXX_LIBRARY}
    debug ${GUROBI_CXX_DEBUG_LIBRARY}
  )
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} ${GUROBI_LIBRARY})

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  include(FeatureSummary)
  feature_summary(WHAT ALL)
endif()
