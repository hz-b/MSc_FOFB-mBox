#include "adc.h"

#include <iostream>

#include "dma.h"
#include "rfmdriver.h"

ADC::ADC(RFMDriver *driver, DMA *dma)
    : m_driver(driver)
    , m_dma(dma)
{}

ADC::~ADC()
{

}

int ADC::read()
{
    RFM2GEVENTINFO eventInfo;              // Info about received interrupts 
    eventInfo.Event = ADC_EVENT;           // We'll wait on this interrupt
    eventInfo.Timeout = ADC_TIMEOUT;       // We'll wait this many milliseconds

    /* Wait on an interrupt from the other Reflective Memory board */
    if (this->waitForEvent( eventInfo ))
        return 1;
    RFM2G_NODE  otherNodeId = eventInfo.NodeId;
    
    t_status *dmaStatus = m_dma->status();
    if ( dmaStatus == NULL )
        return 1;
    dmaStatus->loopPos = eventInfo.ExtendedInfo;

    /* Now read data from the other board from BPM_MEMPOS */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    int data_size = ADC_BUFFER_SIZE  *sizeof( RFM2G_INT16 );

    if (data_size  < threshold) {
        // use PIO transfer
        if (m_driver->read(ADC_MEMPOS + (dmaStatus->loopPos * data_size),
            (void*) m_buffer, data_size )) {
            return 1;
        }
    }
    else {
        if (m_driver->read(ADC_MEMPOS + (dmaStatus->loopPos * data_size),
            (void*) m_dma->memory(), data_size )) {
            return 1;
        }
        
        RFM2G_INT16 * src = (RFM2G_INT16*) m_dma->memory(); 
        for (int i = 0 ; i < ADC_BUFFER_SIZE ; ++i)
        {
            m_buffer[i] = src[i];
        }
    }
    
    return 0;
}

RFM2G_STATUS ADC::waitForEvent(RFM2GEVENTINFO eventInfo)
{
    RFM2G_STATUS res;
    if (res = m_driver->clearEvent( eventInfo.Event ))
        std::cout << "in clearEvent, error: " << res << std::endl;
        return res;
    if (res = m_driver->enableEvent(  eventInfo.Event ))
        std::cout << "in enableEvent, error: " << res << std::endl;
        return res;

    return m_driver->waitForEvent( &eventInfo );
}
