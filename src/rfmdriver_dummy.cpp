#include "rfmdriver_dummy.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <chrono>
#include "define.h"

const std::string dummyFile = "dump_rmf.dat";
RFM2G_STATUS RFMDriver::open(char* devicePath)
{
    std::ifstream infile(dummyFile);
    if(infile.good())
        return RFM2G_SUCCESS;
    else
        return RFM2G_NOT_OPEN;
}

RFM2G_STATUS RFMDriver::read(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length)
{
    std::ifstream file;
    file.open(dummyFile, std::ios::in | std::ios::binary);

    // Check that it doesn't overflow the file
    file.seekg (0, std::ios::end);
    int end = file.tellg();
    if (end < offset+length) {
        std::cout << "## ERROR; Size pb ###\n";
        return RFM2G_DRIVER_ERROR;
    }

    // Read file
    file.seekg(offset);
    file.read((char*)buffer, length);
    file.close();

    return RFM2G_SUCCESS;
}

RFM2G_STATUS RFMDriver::write(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length)
{
    std::ofstream file;
    file.open(dummyFile, std::ios::in | std::ios::out | std::ios::binary);

    // Check that it doesn't overflow the file
    file.seekp (0, std::ios::end);
    int end = file.tellp();

    if (end < offset+length) {
        std::cout << "## ERROR; Size pb ###\n";
        return RFM2G_DRIVER_ERROR;
    }

    // Read file
    file.seekp(offset);
    file.write( (char*)buffer, length );
    file.close();

    return RFM2G_SUCCESS;
}

RFM2G_STATUS RFMDriver::getDMAThreshold(RFM2G_UINT32* threshold)
{
    *threshold = 1000000;//m_DMAthreshold;
    return RFM2G_SUCCESS;
}
RFM2G_STATUS RFMDriver::setDMAThreshold(RFM2G_UINT32 threshold)
{
    m_DMAthreshold = threshold;
    return RFM2G_SUCCESS;
}

RFM2G_STATUS RFMDriver::waitForEvent(RFM2GEVENTINFO* eventInfo)
{
    using namespace std::chrono;
    steady_clock::time_point start = steady_clock::now();
    int filepos = 0;
    if (eventInfo->Event == ADC_EVENT) {
        filepos = -1;
    } else if (eventInfo->Event == DAC_EVENT) {
        filepos = -2;
    }

    milliseconds elapsedTime(0);
    milliseconds timeout(eventInfo->Timeout);
    bool eventReceived = false;
    while (elapsedTime < timeout) {
        std::ifstream file;
        file.open(dummyFile, std::ios::in | std::ios::binary);
        file.seekg(filepos, std::ios::end);
        file.read((char*)&eventReceived, 1);
        file.close();
        if (eventReceived) {
            // reset interruption
            std::ofstream ofile;
            ofile.open(dummyFile, std::ios::in | std::ios::out | std::ios::binary);
            ofile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
            ofile.seekp(filepos, std::ios::end);
            bool buffer = false;
            ofile.write((char*)&buffer, 1);
            ofile.close();
            return RFM2G_SUCCESS;
        }

        steady_clock::time_point stop = steady_clock::now();
        elapsedTime = duration_cast<milliseconds>(stop - start);
    }

    return RFM2G_TIMED_OUT;
}
