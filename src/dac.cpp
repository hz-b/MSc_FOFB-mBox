#include "dac.h"

#include "dma.h"
#include "rfmdriver.h"
#include "define.h"

#include <iostream>

DAC::DAC(RFMDriver *driver, DMA *dma)
    : m_driver(driver)
    , m_dma(dma)
{
    std::string IOCsnames[] = {"IOCS15G", "IOCS2G", "IOCS4G", "IOCS6G", "IOCS8G", "IOCS10G", "IOCS12G", "IOCS14G", "IOCS16G", "IOC3S16G"};
    char nodeIds[] =          { 0x02    ,  0x12   ,  0x14   ,  0x16   ,  0x18   ,  0x1A    ,  0x1C    ,  0x1E    ,  0x20    ,  0x21     };
    bool activeNodes[] =      { true    ,  true   ,  true   ,  true   ,  true   ,  true    ,  true    ,  true    ,  true    ,  true     };
    for (int i = 0 ; i < 10 ; i++) {
        m_IOCs[i] =  IOC(nodeIds[i], IOCsnames[i], activeNodes[i]);
    }
}

void DAC::changeStatus(int status)
{
    if (status == Start) {
        std::cout << "starting DACs ... " << std::endl;
    } else if (status == Stop) {
        std::cout << "stopping DACs ...." << std::endl;
    }
    for (int i = 0 ; i < 10 ; i++) {
        if (m_IOCs[i].isActive()) {
            std::cout << "  " << m_IOCs[i].name() << "\t: ";
            if (m_driver->sendEvent( m_IOCs[i].id(), ADC_DAC_EVENT, status))
            { 
                std::cout << "Error " << std::endl; 
            } else {
                std::cout << "successful" << std::endl;
            }
        }
    }
}

int DAC::write(double plane, double loopDir, RFM2G_UINT32* data)
{
    int writeflag = 0;   
    //plane = 4;
    switch((int) plane) {
    case 0: 
        writeflag = (1<<16) | (1<<17) ;
        break;
    case 1:
        writeflag = 1<<16;
        break;
    case 2:
        writeflag = 1<<17;
        break;
    case 3:
        if (loopDir > 0)
            writeflag = (1<<16);
        else 
            writeflag= (1<<17);
        break;
    }
    writeflag |= (1<<19) | (1<<20); //dummy channel dazu

    RFM2G_UINT32 rfm2gCtrlSeq   = m_dma->status()->loopPos + writeflag;
    RFM2G_UINT32 rfm2gMemNumber = rfm2gCtrlSeq & 0x000ffff;
    // cout << setw(3) << rfm2gMemNumber << " "<< DACout[113] <<"\n";

    /* --- start timer --- */
    //t_dac_start.clock();    

    if (m_driver->clearEvent(DAC_EVENT))
        return 1;
    if (m_driver->enableEvent(DAC_EVENT))
        return 1;
    //t_dac_clear.clock();

    /* fill DAC to RFM */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );
    
    int data_size = DAC_BUFFER_SIZE*sizeof(RFM2G_UINT32);
    if (data_size < threshold) {
         // use PIO transfer
        if (m_driver->write(DAC_MEMPOS + (rfm2gMemNumber*data_size), &data, data_size))
            return 1;
    } else {
        RFM2G_INT32 *dst = (RFM2G_INT32*) m_dma->memory(); 
        for (int i = 0 ; i < DAC_BUFFER_SIZE ; ++i) {
            dst[i] = data[i];
        }
 
        if (m_driver->write(DAC_MEMPOS + (rfm2gMemNumber*data_size),
                            (void*) m_dma->memory(), data_size)) {
            return 1;
        }
    }

 //   t_dac_write.clock();

    //t_adc_start.wait(450000); // sleep 450 us
    //cout << " write "<< t_dac_write.tv_sec << ","<<t_dac_write.tv_nsec <<" \n";

    //usleep(300);

    /* tell IOC to work */
    if (m_driver->sendEvent(RFM2G_NODE_ALL, DAC_EVENT, (RFM2G_INT32) rfm2gCtrlSeq))
        return 1;
    //t_dac_send.clock();    

    /* wait for atleast one ack. */
    RFM2GEVENTINFO EventInfo;
    EventInfo.Event   = DAC_EVENT;    /* We'll wait on this interrupt */
    EventInfo.Timeout = DAC_TIMEOUT;  /* We'll wait this many milliseconds */
    if (m_driver->waitForEvent( &EventInfo )) 
        return 1;

    // stop timer
    //t_dac_stop.clock();    

    return 0;
}