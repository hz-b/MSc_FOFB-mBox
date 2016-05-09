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
#include "logger/messenger.h"

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
    Logger::Logger() << "mBox exited";
}

void mBox::init(const char* deviceName, const bool weightedCorr)
{
    m_currentState = State::Preinit;
    m_mBoxStatus = Status::Idle;
    RFM2GHANDLE RFM_handle = 0;
    m_driver = new RFMDriver(RFM_handle);
    this->initRFM( deviceName );

    m_dma = new DMA();
    if ( int res = m_dma->init(m_driver) )
    {
        Logger::error(_ME_) << "DMA Error .... Quit";
        exit(res);
    }
    Logger::Logger logger;
    logger.setRFM(m_driver);

    if (!m_inputFile.empty()) { // inputFile => Experiment mode
        m_handler = new MeasureHandler(m_driver, m_dma, weightedCorr, m_inputFile);
    } else {
        m_handler = new CorrectionHandler(m_driver, m_dma, weightedCorr);
    }
}

void mBox::startLoop()
{
    Logger::Logger() << "mBox idle";
    for(;;) {
        m_driver->read(CTRL_MEMPOS, &m_mBoxStatus, 1);

#ifdef DUMMY_RFM_DRIVER
        // In case of DUMMY_RFM_DRIVER we'll never get an order from the cBox.
        m_mBoxStatus = Status::Running;
#endif
        if (m_mBoxStatus == Status::RestartedThing) {
            std::cout << "  !!! MDIZ4T4R was restarted !!! ... Wait for initialization \n";
            while (m_mBoxStatus != Status::Idle) {
                m_driver->read(CTRL_MEMPOS , &m_mBoxStatus, 1);
                sleep(1);
            }
            Logger::Logger() << "...Wait for start...";
        }

        // if Idle, don't do anything
        if ((m_mBoxStatus == Status::Idle) && (m_currentState == State::Preinit))
        {
        }

        /**
         * Initialize correction
         */
        if ((m_mBoxStatus == Status::Running) && (m_currentState == State::Preinit)) {
            m_handler->init();
            std::this_thread::sleep_for(std::chrono::nanoseconds(4000000));
            m_currentState = State::Initialized;

            std::cout << "Status: mBox running \n";
            Logger::Logger() << "mBox running";
        }

        /**
         * Read and correct
         */
        if ((m_mBoxStatus == Status::Running) && (m_currentState == State::Initialized)) {
            if (int errornr = m_handler->make()) {
                Logger::postError(errornr);
                m_currentState = State::Error;
                Logger::error(_ME_) <<  "error: " << Logger::errorMessage(errornr);
                std::cout << "Status: mBox in ERROR \n";
            }

            // Write the status
            m_driver->write(STATUS_MEMPOS, m_dma->status(), sizeof(t_status));
        }

        /**
         * Stop correction
         */
        if ((m_mBoxStatus == Status::Idle) && (m_currentState != State::Preinit)) {
            Logger::Logger() << "Stopped  .....";
            std::cout << "Status: mBox in stopped \n";
	        m_handler->disable();
            m_currentState = State::Preinit;
        }

        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000));
    }
}

void mBox::initRFM(const char* deviceName)
{
    Logger::Logger() << "Init RFM";
    Logger::Logger() << "\tRFM Handle : " << m_driver->handle();

    if (m_driver->open(const_cast<char*>(deviceName))) {
        Logger::error(_ME_) << "\tCan't open " << deviceName << '\n' ;
        Logger::error(_ME_) << "\tExit fron initRFM()";
        exit(1);
    }

    RFM2G_NODE nodeId(0);
    if (m_driver->nodeId(&nodeId)) {
        Logger::error(_ME_) << "\tCan't get Node Id";
        exit(1);
    }
    Logger::Logger() << "\tRFM Node Id : " << nodeId;

    Messenger::messenger.startServing();
}

void mBox::parseArgs(int argc, char* argv[])
{
    std::string startflag = "";
    m_inputFile = "";
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (!arg1.compare("--help")) {
            this->printHelp();
            exit(0);
        } else if (!arg1.compare("--ro")) {
            READONLY = true;
            startflag = " [READ-ONLY VERSION]";
        } else if (!arg1.compare("--rw")) {
            READONLY = false;
        } else if (!arg1.compare("--experiment")) {
            if (argc >= 3) {
                READONLY = false;
                m_inputFile = argv[2];
                if (std::ifstream(m_inputFile).good()) {
                    startflag = "[EXPERIMENT MODE] FILE = " + m_inputFile;
                } else {
                    std::cout << "ERROR: " << m_inputFile << " is not a valid file\n\n";
                    startError();
                }
            } else {
                startError();
            }
        } else {
            startError();
        }
    } else {
        startError();
    }
    Logger::Logger logger;
    for (int i=1; i < argc ; i++) {
        if (!std::string(argv[i]).compare("--debug")) {
            logger.setDebug(true);
        } else if (!std::string(argv[i]).compare("--logport")) {
            if ((i+1 < argc) && atoi(argv[i+1]) < 65535 && atoi(argv[i+1]) > 1000) {
                logger.setPort(atoi(argv[i+1]));
            } else {
                std::cout << "A port should be given (1000 to 65535)\n";
                exit(-1);
            }
        } else if (!std::string(argv[i]).compare("--queryport")) {
            if ((i+1 < argc) && atoi(argv[i+1]) < 65535 && atoi(argv[i+1]) > 1000 && atoi(argv[i+1]) != logger.port()) {
                Messenger::messenger.setPort(atoi(argv[i+1]));
            } else {
                std::cout << "A port should be given (1000 to 65535), different from logport.\n";
                exit(-1);
            }
        }
    }
    std::string startMessage = "Starting the mBox " + startflag;
    std::cout << std::string(startMessage.size(),'=') << '\n'
              << startMessage << '\n'
              << std::string(startMessage.size(),'=') << '\n' << '\n';
    Logger::Logger() << startMessage;
}

void mBox::startError()
{
    std::cout << "=== mbox (2015-2016) ===\n";
    std::cout << "One argument is expected: --ro, --rw.\n";
    std::cout << "Or two arguments expected: --experiment <FILE>.\n";
    std::cout << "\n";
    std::cout << "See --help for more help.\n\n";

    exit(-1);
}

void mBox::printHelp()
{
    std::cout << "=== mbox (2015-2016) ===\n"
              << "Use:\n"
              << "mbox --ro\n"
              << "     Read only version: just reads the RFM and calculates\n"
              << "     the correction, don't write it back.\n"
              << "mbox --rw\n"
              << "     Read-write version: reads the RFM, calculates the\n"
              << "     correction and write it on the RFM.\n"
              << "mbox --experiment <FILENAME>\n"
              << "     Read-write version for experiments: read the file <FILENAME>\n"
              << "     to know which values to create.\n\n"
              << "Other arguments (to append):\n"
              << "--debug\n"
              << "     Print the logs on the the stderr.\n"
              << "--logport <PORT>\n"
              << "     Which port the log publisher should use.\n"
              << "--queryport <PORT>\n"
              << "     Which port the query messenger should use.\n\n";
}
