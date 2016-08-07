/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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

/**
 * @brief Status defined by the cBox
 */
enum class Status
{
    Idle = 0,
    Running = 1,
    RestartedThing = 33,
};

/**
 * @brief State of the mBox (the mBox is a state machine)
 */
enum class State
{
    Preinit = 0,
    Initialized = 1,
    Error = 2
};

/**
 * @class mBox
 * @brief Main class: manages the full app
 */
class mBox
{
public:

    /**m_runningState
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
     * @brief Show a small help text and quits. Used when the program is called with wrong arguments.
     */
    void startError();

    /**
     * @brief Print the full help text.
     */
    void printHelp();

    /**
     * @brief Input file: used only in --experiment mode.
     */
    std::string m_inputFile;

    /**
     * @brief Current state of the mBox (state machine).
     */
    State m_currentState;

    /**
     * @brief Status of the mBox (Running? Idle? ...).
     */
    Status m_mBoxStatus;
    DMA *m_dma;
    Handler *m_handler;
    RFMDriver *m_driver;
};

#endif // MBOX_H
