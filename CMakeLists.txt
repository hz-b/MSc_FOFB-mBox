cmake_minimum_required (VERSION 2.8)
project(mBox++)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules )

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Rfm2g REQUIRED)
include_directories(${RFM2G_INCLUDE_DIRS})

find_package(ZeroMQ REQUIRED)
include_directories(${ZEROMQ_INCLUDE_DIRS})

find_package(Threads REQUIRED)

find_package(LibRT REQUIRED)
include_directories(${LIBRT_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)


# DUMMY_RFM_DRIVER
#      1 if on a computer which doesn't support the driver
#      0 if the computer supports the driver
#
add_definitions(-DDUMMY_DRIVER=0)
if (DUMMY_DRIVER)
    set(DUMMY_RFM_DRIVER true)
    message(STATUS "Using dummy RFM driver")
else()
    set(DUMMY_RFM_DRIVER false)
    message(STATUS "Using real RFM driver")
endif()

CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/config.h.cmake ${CMAKE_SOURCE_DIR}/src/config.h )

include_directories(src)
add_subdirectory(src)
