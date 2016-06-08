set(SOURCES main.cpp
            adc.cpp
            dac.cpp
            dma.cpp
            mbox.cpp
            error.cpp
            rfm_helper.cpp
            handlers/handler.cpp
            handlers/correction/correctionhandler.cpp
            handlers/correction/correctionprocessor.cpp
            handlers/correction/dynamic10hzcorrectionprocessor.cpp
            handlers/measures/measurehandler.cpp
            modules/timers.cpp
            modules/zmq/logger.cpp
            modules/zmq/extendedmap.cpp
            modules/zmq/messenger.cpp
            modules/zmq/zmqext.cpp
)

if (${DUMMY_RFM_DRIVER})
    set(SOURCES ${SOURCES} rfmdriver_dummy.cpp)
endif()

add_executable(mbox ${SOURCES})

target_link_libraries (mbox ${ARMADILLO_LIBRARIES}
                            ${ZEROMQ_LIBRARIES}
                            ${RFM2G_LIBRARIES}
                            ${PYTHON_LIBRARY}
                            ${NUMPY_LIBRARY}
                            ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS mbox DESTINATION bin)
