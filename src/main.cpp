#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>
#include <string>

#include "define.h"
#include "mbox.h"
#include "logger/logger.h"
#include "logger/messenger.h"

extern "C" void openblas_set_num_threads(int num_threads);

bool READONLY;

// Must be started first to be deleted last.
zmq::context_t context(1);
zmq_ext::socket_t logSocket(context, ZMQ_PUB /*zmq::socket_type::pub*/);

// Must be static so that exit() do a proper deletion.
static mBox mbox;

namespace Messenger {
    Messenger messenger(context);
}

/**
 * @brief Function called on CTRL+C.
 *
 * Exit the program with output 0.
 */
void SIGINT_handler(int signum)
{
    std::cout << "Quit mBox...\n";
    exit(0);
}

/**
 * @brief Small help text printed when the program is called with wrong arguments.
 */
void startError()
{
    std::cout << "=== mbox (2015-2016) ===\n";
    std::cout << "One argument is expected: --ro, --rw.\n";
    std::cout << "Or two arguments expected: --experiment <FILE>.\n";
    std::cout << "\n";
    std::cout << "See --help for more help.\n\n";

    exit(-1);
}

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    std::string startflag = "";
    std::string experimentFile = "";
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (!arg1.compare("--help")) {
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

            exit(0);
        } else if (!arg1.compare("--ro")) {
            READONLY = true;
            startflag = " [READ-ONLY VERSION]";
        } else if (!arg1.compare("--rw")) {
            READONLY = false;
        } else if (!arg1.compare("--experiment")) {
            if (argc == 3) {
                READONLY = false;
                experimentFile = argv[2];
                startflag = " [EXPERIMENT MODE]\n FILE = " + experimentFile;
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

    Logger::setSocket(&logSocket);

    std::cout << "=============\n"
              << "starting MBox" << startflag << '\n'
              << "=============\n" << std::flush;

    char devicename[] = "/dev/rfm2g0";
    bool weigthedCorr = true;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    Logger::Logger() << "Starting the MBox..." << startflag;

    mbox.init(devicename, weigthedCorr, experimentFile);

    signal(SIGINT, SIGINT_handler);
    mbox.startLoop();

    return 0;
}

