#include "mbox.h"

#include <iostream>
#include <chrono>
#include <thread>

#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "rfmdriver.h"
#include "rfm_helper.h"
#include "handlers/correction/correctionhandler.h"
#include "handlers/measures/measurehandler.h"

#include "logger/logger.h"


mBox::mBox()
{
}

void mBox::init(char *deviceName, bool weightedCorr, std::string inputFile)
{
    m_runningState = Preinit;
    m_runningStatus = Idle;

    RFM2GHANDLE RFM_handle = 0;
    m_driver = new RFMDriver(RFM_handle);

    this->initRFM( deviceName );

    m_dma = new DMA(m_driver);
    if ( int res = m_dma->init() )
    {
        std::cout << "DMA Error .... Quit" << std::endl;
        exit(res);
    }
    m_rfmHelper = new RFMHelper(m_driver, m_dma);
    Logger::logger.init(m_driver, m_dma);

    if (!inputFile.empty()) { // inputFile => Experiment mode
        m_handler = new MeasureHandler(m_driver, m_dma, weightedCorr, inputFile);
    } else {
        m_handler = new CorrectionHandler(m_driver, m_dma, weightedCorr);
    }
}


mBox::~mBox()
{
std::cout << "deleted nbox"<<std::endl;
    delete m_handler,
           m_rfmHelper,
           m_dma,
           m_driver;
}

void mBox::startLoop()
{
    for(;;) {
        m_driver->read(CTRL_MEMPOS, &m_runningStatus, 1);
        m_runningStatus = Running; // HACK
        if (m_runningStatus == 33) {
            std::cout << "  !!! MDIZ4T4R was restarted !!! ... Wait for initialization " << std::endl;
            while (m_runningStatus != Idle) {
                m_driver->read(CTRL_MEMPOS , &m_runningStatus , 1);
                sleep(1);
            }
            //std::cout << "...Wait for start..." << std::endl;
            Logger::logger << "...Wait for start..." /*<< Logger::end */;
        }

        // if Idle, don't do anything
        if ((m_runningStatus == Idle) && (m_runningState == Preinit)) {}

        /**
         * Initialize correction
         */
        if ((m_runningStatus == Running) && (m_runningState == Preinit)) {
            m_handler->init(150, 600);
            m_runningState = Initialized;

            std::cout << "RUN RUN RUN .... " << std::endl;
        }

        /**
         * Read and correct
         */
        if ((m_runningStatus == Running) && (m_runningState == Initialized)) {
            if (int errornr = m_handler->make()) {
                Logger::postError(errornr);
                m_runningState = Error;
                std::cout << "error: " << errornr << std::endl;
            }

            // Write the status
            m_driver->write(STATUS_MEMPOS, m_dma->status(), sizeof(t_status));
        }

        /**
         * Stop correction
         */
        if ((m_runningStatus == Idle) && (m_runningState != Preinit)) {
            std::cout << "Stopped  ....." << std::endl << std::flush;
	    m_handler->disable();
            m_runningState = Preinit;
        }

        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000));
    }
}

void mBox::initRFM(char *deviceName)
{
    std::cout << "Init RFM" << std::endl;
    std::cout << "\tRFM Handle : " << m_driver->handle() << std::endl;

    if (m_driver->open(deviceName)) {
        std::cout << "\tCan't open " << deviceName << std::endl;
        std::cout << "\tExit fron initRFM()" << std::endl;
        exit(1);
    }

    RFM2G_NODE nodeId(0);
    if (m_driver->nodeId(&nodeId)) {
        std::cout << "\tCan't get Node Id" << std::endl;
        exit(1);
    }
    std::cout << "\tRFM Node Id : " << nodeId << std::endl;
}

