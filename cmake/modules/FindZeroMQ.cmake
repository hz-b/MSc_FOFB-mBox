# - Try to find libzmq
# Once done, this will define
#
#  ZeroMQ_FOUND - system has libzmq
#  ZeroMQ_INCLUDE_DIRS - the libzmq include directories
#  ZeroMQ_LIBRARIES - link these to use libzmq

include(LibFindMacros)

IF (UNIX)
        # Use pkg-config to get hints about paths
        libfind_pkg_check_modules(ZeroMQ_PKGCONF libczmq)

        # Include dir
        find_path(ZEROMQ_INCLUDE_DIRS
          NAMES zmq.h
          PATHS ${ZEROMQ_ROOT}/include ${ZeroMQ_PKGCONF_INCLUDE_DIRS}
        )

        # Finally the library itself
        find_library(ZEROMQ_LIBRARIES
          NAMES zmq
          PATHS ${ZEROMQ_ROOT}/lib ${ZeroMQ_PKGCONF_LIBRARY_DIRS}
        )
ELSEIF (WIN32)
        find_path(ZEROMQ_INCLUDE_DIRS
          NAMES zmq.h
          PATHS ${ZEROMQ_ROOT}/include ${CMAKE_INCLUDE_PATH}
        )
        # Finally the library itself
        find_library(ZEROMQ_LIBRARIES
          NAMES libzmq
          PATHS ${ZEROMQ_ROOT}/lib ${CMAKE_LIB_PATH}
        )
ENDIF()

libfind_process(ZeroMQ) 
