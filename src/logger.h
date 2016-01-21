#ifndef LOGGER_H
#define LOGGER_H

#include "rfm_helper.h"
#include <string>
#include <iostream>

class RFMDriver;
class DMA;
namespace Logger {
    
class Logger
{
public:
    Logger() : m_rfmHelper(NULL, NULL) {};
    void init(RFMDriver *driver, DMA *dma) { m_rfmHelper = RFMHelper(driver, dma); }
    static void sendMessage(std::string message)
    {
  //      static ofstream fout("log");
        std::cout << message << std::endl;;
    }
    static void record(std::string message)
    {
//static ofstream fout("log");
        std::cout << message << std::endl;;
    }
    
    void sendMessage(const char* message, const char *error)
    {
        if (READONLY) {
            std::cout << "Message: " << message;
            if (error)
                std::cout << " Error: " << error;
            std::cout << std::endl;
        } else {
            m_rfmHelper.sendMessage(message, error);
        }
    }
private:
    RFMHelper m_rfmHelper;
};


extern Logger logger;

inline void postError(unsigned int errornr)
{
    switch (errornr) {
    case 0:
        return;
        break;
    case FOFB_ERROR_ADC:
        logger.sendMessage( "FOFB error", "ADC Timeout");
        break;
    case FOFB_ERROR_DAC:
        logger.sendMessage( "FOFB error", "DAC Problem");
        break;
    case FOFB_ERROR_CM100:
        logger.sendMessage( "FOFB error", "To much to correct");
        break;
    case FOFB_ERROR_NoBeam:
        logger.sendMessage( "FOFB error", "No Current");
        break;
    case FOFB_ERROR_RMS:
        logger.sendMessage( "FOFB error", "Bad RMS");
        break;
    default:
        logger.sendMessage( "FOFB error", "Unknown Problem");
        break;
    }
}

}

#endif // LOGGER_H
