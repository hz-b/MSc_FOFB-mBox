
set(SOURCES main.cpp
            adc.cpp
            dac.cpp
            dma.cpp
            mbox.cpp
            correctionhandler.cpp
            rfm_helper.cpp
    )

if (${DUMMY_RFM_DRIVER})
    set(SOURCES ${SOURCES} rfmdriver_dummy.cpp)
endif()


add_executable(mbox ${SOURCES})

target_link_libraries (mbox ${ARMADILLO_LIBRARIES}
                            ${ZEROMQ_LIBRARIES}
                            ${RFM2G_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT}
                            ${LIBRT_LIBRARIES}
                        )