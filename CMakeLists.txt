cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME sOPT)
set(TARGET_NAME sOPT)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()
option(SOPT_BUILD_EXAMPLES "Build sOPT examples" ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(${TARGET_NAME} INTERFACE)
add_library(${TARGET_NAME}::${TARGET_NAME} ALIAS ${TARGET_NAME})

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_23)
target_compile_options(${TARGET_NAME} INTERFACE
    $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-O3>
)
target_compile_definitions(${TARGET_NAME} INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
)
# target_compile_options(${TARGET_NAME} INTERFACE
#     $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-march=native>
# )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

target_include_directories(${TARGET_NAME} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} INTERFACE Eigen3::Eigen)

# examples ---------------------------------------------------------------------
if(SOPT_BUILD_EXAMPLES)
    set(SRC_NAME sandbox)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 

    set(SRC_NAME gradient_descent)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 

    set(SRC_NAME newton)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 

    set(SRC_NAME bfgs)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 

    set(SRC_NAME lbfgs)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 

    set(SRC_NAME step_size)
    add_executable(${SRC_NAME} examples/${SRC_NAME}.cpp)
    target_link_libraries(${SRC_NAME} PRIVATE ${TARGET_NAME}::${TARGET_NAME}) 
endif()

# tests ------------------------------------------------------------------------

