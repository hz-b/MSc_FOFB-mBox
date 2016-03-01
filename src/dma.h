#ifndef DMA_H
#define DMA_H

#include "define.h"

class RFMDriver;

class DMA
{
public:
    explicit DMA();
    ~DMA();
    int init(RFMDriver *driver);
    volatile char* memory() { return m_memory; };
    t_status* status() { return m_status; };

private:
    /**
     * DMA Buffer pointer = pDmaCard
     */
    volatile char *m_memory;
    t_status *m_status;
};

#endif // DMA_H
