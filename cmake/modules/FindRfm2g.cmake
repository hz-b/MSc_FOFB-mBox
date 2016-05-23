# - Try to find RFM2g headers and libraries
#
# It defines the followng variables
#  RFM2G_INCLUDE_DIRS - include directories for Armadillo
#  RFM2G_LIBRARY_DIRS - library directories for Armadillo (normally not used!)
#  RFM2G_LIBRARIES    - libraries to link against

# Tell the user project where to find our headers and libraries

find_library (RFM2G_LIBRARIES
    NAMES librfm2g.a
    HINTS /usr/lib/rfm2g
          /usr/lib64/rfm2g
          /usr/local/lib/rfm2g
          /usr/local/lib64/rfm2g
          /usr/lib/x86_64-linux-gnu/rfm2g
    )

find_path (RFM2G_INCLUDE_DIR
    NAMES rfm2g_api.h 
    HINTS /usr/lib64/rfm2g
          /usr/include/rfm2g 
    )

set(RMF2G_VERSION_FILE ${RFM2G_INCLUDE_DIR}/rfm2g_version.h)

## Extract the version 
# from line #define RFM2G_PRODUCT_VERSION "R9.000"
execute_process (
    COMMAND grep "#define RFM2G_PRODUCT_VERSION"
    COMMAND cut -d\"  -f2
    RESULT_VARIABLE VERSION_MINOR_RESULT
    OUTPUT_VARIABLE RFM2G_VERSION
    INPUT_FILE ${RMF2G_VERSION_FILE}
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if (${CMAKE_VERSION} VERSION_GREATER 2.8.1)
    find_package_handle_standard_args (RFM2g
        REQUIRED_VARS RFM2G_LIBRARIES RFM2G_INCLUDE_DIR
        VERSION_VAR RFM2G_VERSION)
else (${CMAKE_VERSION} VERSION_GREATER 2.8.1)
    find_package_handle_standard_args (RFM2g
        DEFAULT_MSG RFM2G_LIBRARIES RFM2G_INCLUDE_DIR)
endif (${CMAKE_VERSION} VERSION_GREATER 2.8.1)

if (RFM2G_FOUND)
    mark_as_advanced (RFM2G_FOUND RFM2G_VERSION 
        RFM2G_LIBRARIES RFM2G_INCLUDE_DIR)
else (RFM2G_FOUND)
    message (FATAL_ERROR "Could not find the RFM2G libraries! Please install the libraries and headers or check that I'm looking to the right places.")
endif (RFM2G_FOUND)

