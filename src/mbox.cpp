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
    : m_dma(NULL)
    , m_driver(NULL)
    , m_handler(NULL)
{
}

mBox::~mBox()
{
    delete m_handler,
           m_dma,
           m_driver;
    Logger::log() << "mBox exited" << Logger::flush;
}

void mBox::init(char *deviceName, bool weightedCorr, std::string inputFile)
{
    m_runningState = Preinit;
    m_runningStatus = Idle;
    RFM2GHANDLE RFM_handle = 0;
    m_driver = new RFMDriver(RFM_handle);
    this->initRFM( deviceName );

    m_dma = new DMA();
    if ( int res = m_dma->init(m_driver) )
    {
        Logger::error(_ME_) << "DMA Error .... Quit" << Logger::flush;
        exit(res);
    }
    Logger::logger.setRFM(m_driver);

    if (!inputFile.empty()) { // inputFile => Experiment mode
        m_handler = new MeasureHandler(m_driver, m_dma, weightedCorr, inputFile);
    } else {
        m_handler = new CorrectionHandler(m_driver, m_dma, weightedCorr);
    }
}

void mBox::startLoop()
{
    Logger::log() << "mBox idle" << Logger::flush;
    for(;;) {
        m_driver->read(CTRL_MEMPOS, &m_runningStatus, 1);

#ifdef DUMMY_RFM_DRIVER
        m_runningStatus = Running; // HACK
#endif
        if (m_runningStatus == 33) {
            std::cout << "  !!! MDIZ4T4R was restarted !!! ... Wait for initialization \n";
            while (m_runningStatus != Idle) {
                m_driver->read(CTRL_MEMPOS , &m_runningStatus , 1);
                sleep(1);
            }
            Logger::log() << "...Wait for start..." << Logger::flush;
        }

        // if Idle, don't do anything
        if ((m_runningStatus == Idle) && (m_runningState == Preinit))
        {
        }

        /**
         * Initialize correction
         */
        if ((m_runningStatus == Running) && (m_runningState == Preinit)) {
            m_handler->init();
            m_runningState = Initialized;

            std::cout << ".... RUNNING .... \n";
            Logger::log() << "mBox running" << Logger::flush;
        }

        /**
         * Read and correct
         */
        if ((m_runningStatus == Running) && (m_runningState == Initialized)) {
            if (int errornr = m_handler->make()) {
                Logger::postError(errornr);
                m_runningState = Error;
                Logger::error(_ME_) <<  "error: " << errornr << Logger::flush;
            }

            // Write the status
            m_driver->write(STATUS_MEMPOS, m_dma->status(), sizeof(t_status));
        }

        /**
         * Stop correction
         */
        if ((m_runningStatus == Idle) && (m_runningState != Preinit)) {
            Logger::log() << "Stopped  ....." << Logger::flush;
	        m_handler->disable();
            m_runningState = Preinit;
        }

        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000));
    }
}

void mBox::initRFM(char *deviceName)
{
    Logger::log() << "Init RFM" << Logger::flush;
    Logger::log() << "\tRFM Handle : " << m_driver->handle() << Logger::flush;

    if (m_driver->open(deviceName)) {
        Logger::error(_ME_) << "\tCan't open " << deviceName << '\n' ;
        Logger::error(_ME_) << "\tExit fron initRFM()" << Logger::flush;
        exit(1);
    }

    RFM2G_NODE nodeId(0);
    if (m_driver->nodeId(&nodeId)) {
        Logger::error(_ME_) << "\tCan't get Node Id" << Logger::flush;
        exit(1);
    }
    Logger::log() << "\tRFM Node Id : " << nodeId << Logger::flush;
}

