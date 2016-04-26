#include "rfmdriver_dummy.h"

#include <fstream>
#include <iostream>
#include <cstring>

RFM2G_STATUS RFMDriver::open(char* devicePath)
{
    std::ifstream infile("dump_rmf.dat");
    if(infile.good())
        return RFM2G_SUCCESS;
    else
        return RFM2G_NOT_OPEN;
}

RFM2G_STATUS RFMDriver::read(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length)
{
    std::ifstream inFile;
    inFile.open( "dump_rmf.dat", std::ios::in | std::ios::binary );

    // Check that it doesn't overflow the file
    inFile.seekg (0, std::ios::end);
    int end = inFile.tellg();
    if (end < offset+length) {
        std::cout << "## ERROR; Size pb ###\n";
        return RFM2G_DRIVER_ERROR;
    }

    // Read file
    inFile.seekg(offset);
    inFile.read( (char*)buffer, length );
    inFile.close();

    return RFM2G_SUCCESS;
}

RFM2G_STATUS RFMDriver::write(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length)
{
    std::ofstream outFile;
    outFile.open( "dump_rmf.dat", std::ios::out | std::ios::binary | std::ios::app );

    // Check that it doesn't overflow the file
    outFile.seekp (0, std::ios::end);
    int end = outFile.tellp();

    if (end < offset+length) {
        std::cout << "## ERROR; Size pb ###\n";
        return RFM2G_DRIVER_ERROR;
    }

    // Read file
    outFile.seekp(offset);
    outFile.write( (char*)buffer, length );
    outFile.close();

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
