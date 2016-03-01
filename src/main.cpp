#include <iostream>
#include <csignal>
#include <string>

#include "define.h"
#include "mbox.h"

extern "C" void openblas_set_num_threads(int num_threads);

bool READONLY;

static mBox mbox;  // Must be static so that exit() do a proper deletion.

void SIGINT_handler(int signum)
{
    std::cout << std::endl << "Quit mBox...." << std::endl;
    exit(0);
}

void startError()
{
    std::cout << "=== mbox (2015-2016) ===" << std::endl;
    std::cout << "One argument is expected: --ro, --rw." << std::endl;
    std::cout << "Or two arguments expected: --experiment <FILE>." << std::endl;
    std::cout << std::endl;
    std::cout << "See --help for more help." << std::endl << std::endl;;

    exit(-1);
}

int main(int argc, char *argv[])
{
    std::string startflag = "";
    std::string experimentFile = "";
    std::string arg1 = argv[1];
    if (argc > 1) {
        if (!arg1.compare("--help")) {
            std::cout << "=== mbox (2015-2016) ===" << std::endl;
            std::cout << "Use:"<< std::endl;
            std::cout << "    mbox --ro\t Read only version: just reads the RFM and calculates" << std::endl;
            std::cout << "             \t the correction, don't write it back." << std::endl;
            std::cout << "    mbox --rw\t Read-write version: reads the RFM, calculates the"  << std::endl;
            std::cout << "             \t correction and write it on the RFM." << std::endl;
            std::cout << "    mbox --experiment <FILENAME>" << std::endl;
            std::cout << "             \t Read-write version for experiments: read the file <FILENAME> " << std::endl;
            std::cout << "             \t to know which values to create." << std::endl;
            std::cout << std::endl;

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
    std::cout << "=============" << std::endl
              << "starting MBox" << startflag << std::endl
              << "=============" << std::endl;

    char devicename[] = "/dev/rfm2g0";
    bool weigthedCorr = true;

    mbox.init(devicename, weigthedCorr, experimentFile);
    std::cout << "mBox ready" << std::endl;

    signal(SIGINT, SIGINT_handler);

    mbox.startLoop();

    return 0;
}

