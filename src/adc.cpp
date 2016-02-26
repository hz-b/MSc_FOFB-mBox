#include "adc.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "dma.h"
#include "rfmdriver.h"

ADC::ADC(RFMDriver *driver, DMA *dma)
    : m_driver(driver)
    , m_dma(dma)
    , m_node(ADC_NODE)
{}

ADC::~ADC()
{

}

int ADC::init(int freq, int DAC_freq)
{
    std::cout << "Init ADC" << std::endl;
    RFM2G_INT32 ctrlBuffer[128];
//    short Navr = DAC_freq/freq;
    ctrlBuffer[0] = 512; // RFM2G_LOOP_MAX
    ctrlBuffer[1] = 0;
    ctrlBuffer[2] = 0;
    ctrlBuffer[3] = 2;          // Navr

    //Stop  ADC Sampling (if running)
    if ((freq == 0) || (DAC_freq == 0)) {
        std::cout << "\tADC stop sampling." << std::endl;

        RFM2G_STATUS sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_STOP);
        if (sendEventError) {
            std::cout << "\tCan't stop ADC." << std::endl;
            return 1;
        }
        std::cout << "\tADC should be stopped." << std::endl;

        return 0;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Write ADC CTRL
    std::cout << "\tADC write sampling config" << std::endl;

    RFM2G_UINT32 threshold = 0;
    // see if DMA threshold and buffer are intialized
    m_driver->getDMAThreshold(&threshold);

    int data_size = 512;
    RFM2G_INT32 writeError;
    if (data_size < threshold) {
       // use PIO transfer
       writeError = m_driver->write(0, &ctrlBuffer, 512);
    } else {
       RFM2G_INT32 *dst = (RFM2G_INT32*)m_dma->memory();
       for (int i = 0 ; i < 128 ; i++) {
           dst[i] = ctrlBuffer[i];
       }
       writeError = m_driver->write(0, (void*)m_dma->memory(), data_size);
    }
    if (writeError) {
        std::cout << "  Can't write ADC config" << std::endl;
        return 1;
    }

    // Enable ADC
    std::cout << "\tADC enable sampling" << std::endl;
    RFM2G_STATUS sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_ENABLE);
    if (sendEventError) {
        std::cout << "\tCan't enable ADC" << std::endl;
        return 1;
    }

    // Start Sampling
    std::cout << "\tADC start sampling" << std::endl;
    sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_START);
    if (sendEventError) {
        std::cout << "\tCan't start sampling" << std::endl;
        return 1;
    }
    std::cout << "\tADC should be started" << std::endl;
    return 0;
}


int ADC::read()
{
    RFM2GEVENTINFO eventInfo;              // Info about received interrupts
    eventInfo.Event = ADC_EVENT;           // We'll wait on this interrupt
    eventInfo.Timeout = ADC_TIMEOUT;       // We'll wait this many milliseconds

    // Wait on an interrupt from the other Reflective Memory board
    RFM2G_STATUS waitError = this->waitForEvent(eventInfo);
    if (waitError)
        return 1;

    if ( m_dma->status() == NULL )
        return 1;

    m_dma->status()->loopPos = eventInfo.ExtendedInfo;
    RFM2G_NODE otherNodeId = eventInfo.NodeId;
    
    /* Now read data from the other board from BPM_MEMPOS */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    int data_size = ADC_BUFFER_SIZE  *sizeof( RFM2G_INT16 );

    if (data_size  < threshold) {
        // use PIO transfer
        RFM2G_STATUS readError = m_driver->read(ADC_MEMPOS + ( m_dma->status()->loopPos * data_size),
                                                (void*)m_buffer, data_size);
        if (readError)
            return 1;

    } else {
        RFM2G_STATUS readError = m_driver->read(ADC_MEMPOS + ( m_dma->status()->loopPos * data_size),
                                                (void*) m_dma->memory(), data_size);
        if (readError)
            return 1;

        RFM2G_INT16 *src = (RFM2G_INT16*) m_dma->memory();
        for (int i = 0 ; i < ADC_BUFFER_SIZE ; ++i) {
            m_buffer[i] = src[i];
        }
    }

    // Send an interrupt to the other Reflective Memory board
    RFM2G_STATUS sendEventError = m_driver->sendEvent(otherNodeId, ADC_EVENT, 0);
    if (sendEventError)
        return 1;

    return 0;
}

RFM2G_STATUS ADC::waitForEvent(RFM2GEVENTINFO eventInfo)
{
    RFM2G_STATUS eventError = m_driver->clearEvent(eventInfo.Event);
    if (eventError) {
        std::cout << "in clearEvent, error: " << eventError << std::endl;
        return eventError;
    }
    eventError = m_driver->enableEvent(eventInfo.Event);
    if (eventError) {
        std::cout << "in enableEvent, error: " << eventError << std::endl;
        return eventError;
    }

    RFM2G_STATUS waitForEventError = m_driver->waitForEvent(&eventInfo);
    return waitForEventError;
}
