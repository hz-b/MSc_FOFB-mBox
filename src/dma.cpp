#include "dma.h"

#include <unistd.h>
#include <iostream>
#include <cstdio>

#include "rfmdriver.h"


DMA::DMA(RFMDriver *driver) : m_driver(driver)
{
    m_status = new t_status;
}

DMA::~DMA()
{
    delete m_status,
           m_memory; // Not sure if this one should be deleted
}

int DMA::init()
{
    RFM2G_UINT32 rfm2gSize;
    volatile void *pPioCard = NULL;
    volatile char *pDmaCard = NULL;   // alias MappedDmaBase

    std::cout << "RFM DMA Init " << std::endl;

    m_driver->setDMAThreshold(DMA_THRESHOLD);

    RFM2GCONFIG rfm2gConfig;
    if(m_driver->getConfig(&rfm2gConfig) == RFM2G_SUCCESS) {
        pPioCard = (char*)rfm2gConfig.PciConfig.rfm2gBase;
        rfm2gSize = rfm2gConfig.PciConfig.rfm2gWindowSize;      
    }

    int pageSize = getpagesize();
    unsigned int numPagesDMA = rfm2gSize / (2* pageSize);
    unsigned int numPagesPIO = rfm2gSize / (2* pageSize);
    if((rfm2gSize % pageSize) > 0) {
        std::cout << "Increase PIO and DMA " << std::endl;
        numPagesDMA++;
        numPagesPIO++;
    }

    //numPagesDMA = 100;

    std::cout << "   pPioCard : " << pPioCard << std::endl;
    std::cout << "   rfm2gSize : " << rfm2gSize << std::endl;
    std::cout << "   pageSize  : " << pageSize << std::endl;
    std::cout << "   numPages DMA/PIO : " << numPagesDMA << std::endl;

    RFM2G_STATUS rfmReturnStatus = m_driver->userMemory(
                                        (volatile void **) (&pDmaCard),
                                        (DMAOFF_A | LINUX_DMA_FLAG),
                                        numPagesDMA
                                        );
    if(rfmReturnStatus != RFM2G_SUCCESS) {
        std::printf("doDMA: ERROR: Failed to map card DMA buffer; %s\n",
            m_driver->errorMsg(rfmReturnStatus));
        return -1;
    }
    std::printf("doDMA: SUCCESS: mapped numPagesDMA=%d at pDmaCard=%p\n", numPagesDMA, pDmaCard);
    m_memory = pDmaCard;

    rfmReturnStatus = m_driver->userMemoryBytes(
                            (volatile void **) (&pPioCard),
                            (0x00000000 | LINUX_DMA_FLAG2),
                            rfm2gSize
                            );
    if(rfmReturnStatus != RFM2G_SUCCESS) {
        std::printf("doDMA: ERROR: Failed to map card PIO; %s\n", 
                    m_driver->errorMsg(rfmReturnStatus));

        return -1;
    }
    std::printf("doDMA: Card: PIO memory pointer = 0x%X, Size = 0x%X\n", pPioCard, rfm2gSize);

    return 0;
}
