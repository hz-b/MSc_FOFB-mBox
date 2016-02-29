#ifndef MBOX_H
#define MBOX_H

#include <armadillo>
#include "define.h"

class Handler;
class RFMDriver;
class RFMHelper;
class ADC;
class DAC;
class DMA;

class mBox
{
public:
    explicit mBox();
    void init(char *deviceName, bool weightedCorr,std::string inputFile);
    ~mBox();
    void startLoop();
    void sendMessage(const char* message, const char *error);
private:
    void initRFM(char *deviceName);
    void writeCorrection(RFM2G_UINT32* DACout);
    void postError(unsigned int errornr);
    bool m_readOnly;
    RunState m_runningState;
    RunStatus m_runningStatus;
    DMA *m_dma;
    Handler *m_handler;
    RFMDriver *m_driver;
    RFMHelper *m_rfmHelper;

};

#endif // MBOX_H
