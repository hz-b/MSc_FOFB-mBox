#include "rfmdriver_dummy.h"

#include <fstream>
#include <iostream>
#include <cstring> 

RFM2G_STATUS RFMDriver::open(char *devicePath)
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
    int begin = inFile.tellg();
    inFile.seekg (0, std::ios::end);
    int end = inFile.tellg();
    if ((end - begin) < offset+length) {
        std::cout << "## ERROR; Size pb ###" << std::endl;
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
     return RFM2G_SUCCESS;
}

/*RFM2G_STATUS RFMDriver::getConfig(RFM2GCONFIG *config)
{
    config = m_config;
    return RFM2G_SUCCESS;
};*/
