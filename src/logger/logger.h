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


// https://savingyoutime.wordpress.com/2009/04/21/using-c-stl-streambufostream-to-create-time-stamped-logging-class/
// http://www.horstmann.com/cpp/streams.txt
namespace Logger {

class Logger
{
public:
    explicit Logger(zmq::context_t &context);
    ~Logger();
    void init(RFMDriver *driver, DMA *dma) { m_rfmHelper = RFMHelper(driver, dma); }
    void record(std::string message);

    void sendMessage(std::string message, std::string error="");
    void sendZmq(std::string message);

    template<typename T>
    Logger& operator<< (const T& data)
    {
  /*    if (data == Logger::end)
        {
            std:: cout << m_buffer << std::endl;
        }*/
        std::ostringstream stream;
        stream << data;
        if (!m_buffer.empty()) {
            m_buffer.append(" ");
        }
        m_buffer.append(stream.str());
    };

private:
    RFMHelper m_rfmHelper;
    zmq::socket_t *m_zmqSocket;
    std::thread *m_thread;
    std::string m_buffer;
};


extern Logger logger;

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
