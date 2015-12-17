#ifndef ADC_H
#define ADC_H

#include "define.h"

class RFMDriver;
class DMA;

class ADC
{
public:
    explicit ADC(RFMDriver *driver, DMA *dma);
    ~ADC();
    int read();
    RFM2G_INT16 bufferAt(int id) const { return m_buffer[id]; };
    double waveIndexXAt(int id) const { return m_waveIndexX[id]; };
    double waveIndexYAt(int id) const { return m_waveIndexY[id]; };
    void setWaveIndexX(double *data) { m_waveIndexX = data; };
    void setWaveIndexY(double *data) { m_waveIndexY = data; };

private:
    RFM2G_STATUS waitForEvent(RFM2GEVENTINFO eventInfo);
    DMA *m_dma;
    RFMDriver *m_driver;
    RFM2G_INT16 m_buffer[ADC_BUFFER_SIZE];
    double* m_waveIndexX;
    double* m_waveIndexY;
};

#endif // ADC_H
