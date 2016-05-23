# Taken from: https://github.com/airsim/stdair
# (License: GPL2)
#
# - Try to find ZEROMQ headers and libraries
#
# Usage of this module as follows:
#
#     find_package(ZEROMQ)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  ZEROMQ_ROOT_DIR  Set this variable to the root installation of
#                            ZEROMQ if the module has problems finding 
#                            the proper installation path.
#
# Variables defined by this module:
#
#  ZEROMQ_FOUND              System has ZEROMQ libs/headers
#  ZEROMQ_LIBRARIES          The ZeroMQ libraries
#  ZEROMQ_INCLUDE_DIR        The location of ZeroMQ headers
find_path (ZEROMQ_ROOT_DIR
    NAMES include/zmq.hpp)

find_library (ZEROMQ_LIBRARIES
    NAMES zmq
    HINTS ${ZEROMQ_ROOT_DIR}/lib
          ${ZEROMQ_ROOT_DIR}/x86_64-linux-gnu)
      
find_path (ZEROMQ_INCLUDE_DIR
    NAMES zmq.hpp
    HINTS ${ZEROMQ_ROOT_DIR}/include)

set (ZMQ_CFG_FILE ${ZEROMQ_INCLUDE_DIR}/zmq.h)

# Extract the version major level
execute_process (
    COMMAND grep "#define ZMQ_VERSION_MAJOR"
    COMMAND cut -d\  -f3
    RESULT_VARIABLE VERSION_MAJOR_RESULT
    OUTPUT_VARIABLE ZMQ_VERSION_MAJOR
    INPUT_FILE ${ZMQ_CFG_FILE}
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Extract the version minor level
execute_process (
    COMMAND grep "#define ZMQ_VERSION_MINOR"    
    COMMAND cut -d\  -f3
    RESULT_VARIABLE VERSION_MINOR_RESULT
    OUTPUT_VARIABLE ZMQ_VERSION_MINOR
    INPUT_FILE ${ZMQ_CFG_FILE}
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Extract the version patch level
execute_process (
    COMMAND grep "#define ZMQ_VERSION_PATCH"
    COMMAND cut -d\  -f3
    RESULT_VARIABLE VERSION_PATCH_RESULT
    OUTPUT_VARIABLE ZMQ_VERSION_PATCH
    INPUT_FILE ${ZMQ_CFG_FILE}
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set (ZEROMQ_VERSION "${ZMQ_VERSION_MAJOR}.${ZMQ_VERSION_MINOR}")

##
# Check that the just (above) defined variables are valid paths:
#  - ZEROMQ_LIBRARIES
#  - ZEROMQ_INCLUDE_DIR
# In that case, ZEROMQ_FOUND is set to True.

# Given the way those variables have been calculated, they should
# either be defined or correspond to valid paths. We use the
# find_package_handle_standard_args() CMake macro to have a standard behaviour.
include (FindPackageHandleStandardArgs)

if (${CMAKE_VERSION} VERSION_GREATER 2.8.1)
    find_package_handle_standard_args (ZeroMQ 
        REQUIRED_VARS ZEROMQ_LIBRARIES ZEROMQ_INCLUDE_DIR
        VERSION_VAR ZEROMQ_VERSION)
else (${CMAKE_VERSION} VERSION_GREATER 2.8.1)    
    find_package_handle_standard_args (ZeroMQ
        DEFAULT_MSG ZEROMQ_LIBRARIES ZEROMQ_INCLUDE_DIR)
endif (${CMAKE_VERSION} VERSION_GREATER 2.8.1)

if (ZEROMQ_FOUND)
    mark_as_advanced (ZEROMQ_FOUND ZEROMQ_VERSION 
        ZEROMQ_LIBRARIES ZEROMQ_INCLUDE_DIR)
    #    message (STATUS "Found ZEROMQ version: ${ZEROMQ_VERSION}")
else (ZEROMQ_FOUND)
    message (FATAL_ERROR "Could not find the ZeroMQ libraries! Please install the development-libraries and headers (e.g., 'zeromq-devel' for Fedora/RedHat).")
endif (ZEROMQ_FOUND)

