#ifndef DAC_H
#define DAC_H

#include <string>
#include <vector>

#include "define.h"

class DMA;
class RFMDriver;

class IOC {
public:
    explicit IOC() : IOC(0, "", false) {};
    explicit IOC(int id, std::string name, bool active) { m_id = id; m_name = name; m_active = active; };
    int id() { return m_id; };
    std::string name() { return m_name; };
    bool isActive() { return m_active; };

private:
    int m_id;
    std::string m_name;
    bool m_active;
};

class DAC
{
public:
    enum Status {
        Stop = 1,
        Start = 2
    };
    explicit DAC(RFMDriver *driver, DMA *dma);
    void changeStatus(int status);
    double waveIndexXAt(int id) const { return m_waveIndexX.at(id); };
    double waveIndexYAt(int id) const { return m_waveIndexY.at(id); };
    void setWaveIndexX(std::vector<double> vect) { m_waveIndexX = vect; };
    void setWaveIndexY(std::vector<double> vect) { m_waveIndexY = vect; };
    int write(double plane, double loopDir, RFM2G_UINT32* data);


private:
    DMA *m_dma;
    RFMDriver *m_driver;
    RFM2G_INT16 m_buffer[DAC_BUFFER_SIZE];
    std::vector<double> m_waveIndexX;
    std::vector<double> m_waveIndexY;
    IOC m_IOCs[10];
};

#endif // DAC_H
