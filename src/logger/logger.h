#ifndef LOGGER_H
#define LOGGER_H

#include "rfm_helper.h"

#include <string>
#include <iostream>
#include <sstream>

#include "logger/zmqext.h"

#define _ME_ __PRETTY_FUNCTION__

namespace std { class thread; }
namespace zmq { class socket_t; }
class RFMDriver;
class DMA;


namespace Logger {

struct log_stream_t {
    std::string header;
    std::ostringstream message;
};

class Logger
{
public:
    explicit Logger(zmq::context_t &context);
    ~Logger();
    void init(RFMDriver *driver, DMA *dma) { m_rfmHelper = RFMHelper(driver, dma); }
    void record(std::string message);

    void sendMessage(std::string message, std::string error="");
    void sendZmq(const std::string& header, const std::string& message);

    log_stream_t m_logStream;

private:
    RFMHelper m_rfmHelper;
    zmq_ext::socket_t *m_zmqSocket;
    std::thread *m_thread;
    std::string m_buffer;
};



extern Logger logger;

inline std::ostream& flush(std::ostream & output)
{
    logger.sendZmq(logger.m_logStream.header, logger.m_logStream.message.str());
    logger.m_logStream.header = "";
    logger.m_logStream.message.str("");
}

inline std::ostringstream& log(std::string type)
{
    logger.m_logStream.header = type ;
    return logger.m_logStream.message;
}

inline std::ostream& error(std::string name) {
    return std::cerr << "\x1b[1;33;41m[" << name << "]\x1b[0m ";
}

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
