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
    ~mBox();

    /**
     * @brief Initialiwe the mBox.
     *
     * @param deviceName Name of the RFM device (to be passed to RFMDriver)
     * @param weightedCorr True if the correction is weighted
     * @param inputFile File to use for experiments (eg: Python module)
     */
    void init(char *deviceName, bool weightedCorr,std::string inputFile);

    /**
     * @brief Start the main loop that handle the different events
     */
    void startLoop();

private:

    /**
     * @brief Initialize the RFM.
     * @param deviceName Name of the RFM device
     */
    void initRFM(char *deviceName);

    RunState m_runningState;
    RunStatus m_runningStatus;
    DMA *m_dma;
    Handler *m_handler;
    RFMDriver *m_driver;
};

#endif // MBOX_H
