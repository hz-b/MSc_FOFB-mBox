# - Config file for the Armadillo package
# It defines the following variables
#  RFM2G_INCLUDE_DIRS - include directories for Armadillo
#  RFM2G_LIBRARY_DIRS - library directories for Armadillo (normally not used!)
#  RFM2G_LIBRARIES    - libraries to link against

# Tell the user project where to find our headers and libraries
set(RFM2G_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/rfm2g/include")
set(RFM2G_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/rfm2g/lib")

# Finally the library itself
find_library(RFM2G_LIBRARIES
		NAMES rfm2g
		PATHS ${RFM2G_LIBRARY_DIRS}
	)

