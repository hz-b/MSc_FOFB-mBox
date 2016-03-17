#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>
#include <string>

#include "define.h"
#include "mbox.h"
#include "logger/logger.h"

extern "C" void openblas_set_num_threads(int num_threads);

bool READONLY;

// Must be started first to be deleted last.
namespace Logger {
    zmq::context_t context(1);
    Logger logger(context);
}

// Must be static so that exit() do a proper deletion.
static mBox mbox;

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
            std::cout << "=== mbox (2015-2016) ===\n";
            std::cout << "Use:\n";
            std::cout << "    mbox --ro\t Read only version: just reads the RFM and calculates\n";
            std::cout << "             \t the correction, don't write it back.\n";
            std::cout << "    mbox --rw\t Read-write version: reads the RFM, calculates the\n";
            std::cout << "             \t correction and write it on the RFM.\n";
            std::cout << "    mbox --experiment <FILENAME>";
            std::cout << "             \t Read-write version for experiments: read the file <FILENAME>\n";
            std::cout << "             \t to know which values to create.\n\n";
            std::cout << "Other arguments (to append):\n";
            std::cout << "    --debug  \t Print the logs on the the stderr.\n\n";

            return 0;
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
    if (!std::string(argv[argc-1]).compare("--debug")) {
        Logger::setDebug(true);
    }


    std::cout << "=============\n"
              << "starting MBox" << startflag << '\n'
              << "=============\n" << std::flush;

    char devicename[] = "/dev/rfm2g0";
    bool weigthedCorr = true;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    Logger::log() << Logger::flush;
    Logger::log() << "Starting the MBox...";
    if (!startflag.empty())
        Logger::log() << startflag;
    Logger::log() << Logger::flush;
    
    mbox.init(devicename, weigthedCorr, experimentFile);

    Logger::log() << "mBox ready" << Logger::flush;

    signal(SIGINT, SIGINT_handler);
    mbox.startLoop();

    return 0;
}

