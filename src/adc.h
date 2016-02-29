#ifndef ADC_H
#define ADC_H

#include "define.h"
#include <vector>

class RFMDriver;
class DMA;

class ADC
{
public:
    explicit ADC(RFMDriver *driver, DMA *dma);
    ~ADC();
    int init(int freq, int DAC_freq);
    int read();
    RFM2G_INT16 bufferAt(int id) const { return m_buffer[id]; };
    double waveIndexXAt(int id) const { return m_waveIndexX.at(id); };
    double waveIndexYAt(int id) const { return m_waveIndexY.at(id); };
    void setWaveIndexX(std::vector<double> vect) { m_waveIndexX = vect; };
    void setWaveIndexY(std::vector<double> vect) { m_waveIndexY = vect; };

private:
    RFM2G_STATUS waitForEvent(RFM2GEVENTINFO &eventInfo);
    DMA *m_dma;
    RFMDriver *m_driver;
    RFM2G_INT16 m_buffer[ADC_BUFFER_SIZE];
    std::vector<double> m_waveIndexX;
    std::vector<double> m_waveIndexY;
    int m_node;
};

#endif // ADC_H
