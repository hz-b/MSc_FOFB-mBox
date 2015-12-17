# Try to find librt
# Once done this will define
#
#  LIBRT_FOUND - system has librt
#  LIBRT_INCLUDE_DIR - the librt include directory
#  LIBRT_LIBRARIES - the libraries needed to use librt
#  LIBRT_STATIC_LIBRARIES - static libraries of librt
#
# The following variables control the behaviour of this module:
#
# LIBRT_DIR:         Specify a custom directory where suitesparse is located
#                    libraries and headers will be searched for in
#                    ${LIBRT_DIR}/include and ${LIBRT_DIR}/lib
include(LibFindMacros)

	FIND_PATH(LIBRT_INCLUDE_DIR NAMES time.h 
								PATHS	${LIBRT_DIR}/include/
										/usr/include/
										/usr/local/include/
	)

	FIND_LIBRARY(LIBRT_LIBRARY rt 
				 PATHS	${LIBRT_DIR}/lib/
						/usr/local/lib64/
						/usr/local/lib/
						/usr/lib/i386-linux-gnu/
						/usr/lib/x86_64-linux-gnu/
						/usr/lib64/
						/usr/lib/
	)
	
	FIND_FILE(LIBRT_STATIC_LIBRARIES librt.a
			  PATHS		${LIBRT_DIR}/lib/
						/usr/local/lib64/
						/usr/local/lib/
						/usr/lib/i386-linux-gnu/
						/usr/lib/x86_64-linux-gnu/
						/usr/lib64/
						/usr/lib/
	)
	
	libfind_process(LibRT) 

	if( LIBRT_FOUND )
		set( LIBRT_LIBRARIES ${LIBRT_LIBRARY} )
	endif( LIBRT_FOUND )