//#include "mBox_RO.cc"

#include <iostream>
#include <csignal> 
#include <cstdlib>
#include <cstring>
#include <string>

#include "define.h"
#include "mbox.h"


extern "C" void openblas_set_num_threads(int num_threads);

void SIGINT_handler(int signum)
{
    std::cout << std::endl << "Quit mBox...." << std::endl;
    exit(0); 
}

void startError()
{
    std::cout << "=== mbox (2015-2016) ===" << std::endl;
    std::cout << "One argument is expected: --ro or --rw." << std::endl;
    std::cout << std::endl;
    std::cout << "See --help for more help." << std::endl << std::endl;;
    
    exit(-1);
}

int main(int argc, char *argv[])
{
    std::string startflag = "";

    if (argc == 2) {
        if (!std::strcmp(argv[1], "--help")) {
            std::cout << "=== mbox (2015-2016) ===" << std::endl;
            std::cout << "Use:"<< std::endl;
            std::cout << "    mbox --ro\t Read only version: just reads the RFM and calculates" << std::endl;
            std::cout << "\t\t the correction, don't write it back." << std::endl;
            std::cout << "    mbox --rw\t Read-write version: reads the RFM, calculates the"  << std::endl;
            std::cout << "\t\t correction and write it on the RFM." << std::endl;
            std::cout << std::endl;

            return 0;
        } else if (!std::strcmp(argv[1], "--ro")) {
            READONLY = true;
            startflag = " [READ-ONLY VERSION]";
        } else if (!std::strcmp(argv[1], "--rw")) {
            READONLY = false;
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
    mBox mbox(devicename, weigthedCorr);
    std::cout << "mBox ready" << std::endl;

    signal(SIGINT, SIGINT_handler);

    std::cout << "Wait for start" << std::endl;
    mbox.startLoop();

    return 0;
}

