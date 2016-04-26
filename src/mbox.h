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

    /**
     * @brief Constructor
     */
    explicit mBox();

    /**
     * @brief Destructor
     */
    ~mBox();

    /**
     * @brief Parse the commandline arguments
     */
    void parseArgs(int argc, char* argv[]);

    /**
     * @brief Initialize the mBox.
     *
     * @param deviceName Name of the RFM device (to be passed to RFMDriver)
     * @param weightedCorr True if the correction is weighted
     */
    void init(const char* deviceName, const bool weightedCorr);

    /**
     * @brief Start the main loop that handle the different events
     */
    void startLoop();

private:

    /**
     * @brief Initialize the RFM.
     * @param deviceName Name of the RFM device
     */
    void initRFM(const char* deviceName);

    /**
     * @brief Small help text printed when the program is called with wrong arguments.
     */
    void startError();

    std::string m_inputFile;
    RunState m_runningState;
    RunStatus m_runningStatus;
    DMA *m_dma;
    Handler *m_handler;
    RFMDriver *m_driver;
};

#endif // MBOX_H
