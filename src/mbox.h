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

enum RunStatus
{
    Idle = 0,
    Running = 1,
};

enum RunState
{
    Preinit = 0,
    Initialized = 1,
    Error = 2
};

class mBox
{
public:
    explicit mBox(char *deviceName, bool weightedCorr);
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
    //FS BUMP

};

#endif // MBOX_H
