#include "dma.h"

#include <unistd.h>  // needed for getpagesize()
#include <iostream>

#include "rfmdriver.h"


DMA::DMA()
    : m_memory(NULL)
{
    m_status = new t_status;
}

DMA::~DMA()
{
    delete m_status,
           m_memory; // Not sure if this one should be deleted
}

int DMA::init(RFMDriver *driver)
{
    RFM2G_UINT32 rfm2gSize;
    volatile void *pPioCard = NULL;

    std::cout << "RFM DMA Init \n";

    driver->setDMAThreshold(DMA_THRESHOLD);

    RFM2GCONFIG rfm2gConfig;
    RFM2G_STATUS getConfigError = driver->getConfig(&rfm2gConfig);
    if (getConfigError) {
        pPioCard = (char*)rfm2gConfig.PciConfig.rfm2gBase;
        rfm2gSize = rfm2gConfig.PciConfig.rfm2gWindowSize;
    }

    int pageSize = getpagesize();
    unsigned int numPagesDMA = rfm2gSize / (2* pageSize);
    unsigned int numPagesPIO = rfm2gSize / (2* pageSize);
    if ((rfm2gSize % pageSize) > 0) {
        std::cout << "Increase PIO and DMA \n";
        numPagesDMA++;
        numPagesPIO++;
    }

    std::cout << "\tpPioCard : " << pPioCard << std::endl;
    std::cout << "\trfm2gSize : " << rfm2gSize << std::endl;
    std::cout << "\tpageSize  : " << pageSize << std::endl;
    std::cout << "\tnumPages DMA/PIO : " << numPagesDMA << std::endl;

    RFM2G_STATUS rfmMemoryError = driver->userMemory(
                                        (volatile void **) (&m_memory),
                                        (DMAOFF_A | LINUX_DMA_FLAG),
                                        numPagesDMA
                                        );
    if (rfmMemoryError) {
        std::cout << "doDMA: ERROR: Failed to map card DMA buffer; "
                  << driver->errorMsg(rfmMemoryError)
                  << std ::endl;
        return -1;
    }
    std::cout << "doDMA: SUCCESS: mapped numPagesDMA=" << numPagesDMA
              << " at pDmaCard=" << std::hex << m_memory
              << std::endl;

    rfmMemoryError = driver->userMemoryBytes((volatile void **) (&pPioCard),
                                             (0x00000000 | LINUX_DMA_FLAG2),
                                             rfm2gSize);
    if (rfmMemoryError) {
        std::cout << "doDMA: ERROR: Failed to map card PIO; "
                  << driver->errorMsg(rfmMemoryError)
                  << std::endl;

        return -1;
    }
    std::cout << "doDMA: Card: PIO memory pointer = 0x" << std::hex << pPioCard
              << ", Size = 0x" << std::hex << rfm2gSize
              << std::endl;

    return 0;
}
