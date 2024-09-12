cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(fruits-and-vegetables-image-recognition)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

add_executable(main src/main.cpp
                    src/utils.cpp
                    src/dataset.cpp
                    src/log.cpp
                    include/main.h
                    include/utils.h
                    include/dataset.h
                    include/log.h)

target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/artifacts/resnet50_torch.pth ${CMAKE_CURRENT_BINARY_DIR}/artifacts/resnet50_torch.pth COPYONLY)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)