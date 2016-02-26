#ifndef DMA_H
#define DMA_H

#include "define.h"

class RFMDriver;

class DMA
{
public:
    explicit DMA(RFMDriver *driver);
    ~DMA();
    int init();
    volatile char* memory() { return m_memory; };
    t_status* status() { return m_status; };

private:
    /* DMA Buffer pointer */
    volatile char *m_memory;
    RFMDriver *m_driver;
    t_status *m_status;
};

#endif // DMA_H
