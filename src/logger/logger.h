#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include "define.h"
#include "logger/zmqext.h"
#include "rfmdriver.h"

#define _ME_ __PRETTY_FUNCTION__ /**< Just to save some typing... */

namespace zmq { class socket_t; }
class RFMDriver;
class DMA;

/**
 * @brief Type of Log
 */
enum class LogType {
    None = 0,
    Log = 1,
    Error = 2,
};

/**
 * @brief Type of value
 */
enum class LogValue {
    None = 0,
    BPM,
    CM,
    ADC
};
/**
 * \addtogroup
 */
namespace Logger {

struct log_stream_t {
    log_stream_t(LogType type, std::string s_other = "") : header(type), other(s_other) {}
    LogType header;
    std::ostringstream message;
    std:: string other;
};

/**
 * @class Logger
 *
 * Relies on some static variables (m_debug, m_port, m_socket, m_driver).
 *
 * @note Use it as:
 * \code{.cpp}
 *      Logger::Logger() << "This is an log line";
 * \endcode
 */
class Logger
{
public:
    /**
     * @brief Constructor
     *
     * The ZMQ publisher socket is set here with the default port.
     */
    explicit Logger(LogType type = LogType::Log, std::string other="");

    /**
     * @brief Destructor
     */
    ~Logger();

    /**
     * @brief Set the RFM Helper.
     */
    void setRFM(RFMDriver* driver) { Logger::m_driver = driver; }

    void sendMessage(const std::string& message, const std::string& error="");
    void sendZmq(const std::string& header, const std::string& message, const std::string& other);
    //void sendZmqValue(const std::string& header, const int loopPos, const arma::vec& valueX, const arma::vec& valueY);
    //void sendZmqValue(const std::string& header, const int loopPos, const std::vector<RFM2G_INT16>& value);
    template <typename T>
    void sendZmqValue(const std::string& header, const int loopPos, const std::vector<T>& values)
    {
        if (values.size() == 0) {
            "ERROR -- Tried to send empty values, return";
            return;
        }
        m_zmqSocket->send(header, ZMQ_SNDMORE);
        m_zmqSocket->send(loopPos, ZMQ_SNDMORE);
        std::string type;
        if (typeid(values.at(0).at(0)) == typeid(double)) {
            type = "double";
        } else if (typeid(values.at(0).at(0)) == typeid(short)) {
            type = "short";
        }
        m_zmqSocket->send(type, ZMQ_SNDMORE);

        // Loop until size-1 because last should not have a ZMQ_SNDMORE flag
        for (int i = 0 ; i < values.size() - 1 ; i++) {
            m_zmqSocket->send(values.at(i), ZMQ_SNDMORE);
        }
        m_zmqSocket->send(values.back());
    }

    /**
     * @brief Send message to the RFM.
     * @param message The message
     * @param errornr Error number
     */
    void sendRFM(const std::string& message, const std::string& error);

    /**
     * @brief Set/Unset the debug mode
     */
    void setDebug(const bool debug) { m_debug = debug; }

    /**
     * @brief Set the socket and bind it.
     */
    void setSocket(zmq_ext::socket_t* socket);

    /**
     * @brief Set the port for the publisher
     */
    void setPort(const int port);

    /**
     * @brief Get the publisher port.
     */
    int port() const;

    /**
     * @brief Check if the mbox is in debug mode.
     *
     * @return True if debug mode, else false.
     */
    bool hasDebug() const { return m_debug; }

    /**
     * @brief Acess to m_logStream
     */
    log_stream_t* logStream() { return m_logStream; };

    /**
     * @brief Append to the message buffer.
     *
     * @param value value to add to the buffer.
     * @return A Logger object
     */
    template <typename T> Logger &operator<<(T value) { m_logStream->message << value; return *this;}

private:
    void parseAndSend();

    static RFMDriver* m_driver;

    /**
     * @brief ZMQ Socket used to publish logs, values, errors.
     */
    static zmq_ext::socket_t* m_zmqSocket;

    /**
     * @brief Is the mBox in debug mode?
     */
    static bool m_debug;
    log_stream_t* m_logStream;
    static int m_port;
};


/**
 * @brief Global wrapper to access the setDebug(bool) method of the class Logger.
 */
void setDebug(const bool debug);

/**
 * @brief Global wrapper to access the setSocket(socket_t*) method of the class Logger.
 */
void setSocket(zmq_ext::socket_t* socket);

/**
 * @brief Global wrapper to access the setPort(int) method of the class Logger.
 */
void setPort(const int port);

/**
 * @brief Send a value over ZMQ.
 */
void values(LogValue name, const int loopPos, const arma::vec& valueX, const arma::vec& valueY);

/**
 * @brief Send a value over ZMQ.
 */
void values(LogValue name, const int loopPos, const std::vector<RFM2G_INT16>& value);

template <typename T>
void values(LogValue name, const int loopPos, const std::vector<T> values)
{
    std::string header;
    switch (name) {
    case LogValue::BPM:
        header = "FOFB-BPM-DATA";
        break;
    case LogValue::CM:
        header = "FOFB-CM-DATA";
        break;
    case LogValue::ADC:
        header = "FOFB-ADC-DATA";
        break;
    default:
        std::cout << "ERROR -- Tried to send values of unexpected type. RETURN";
        return;
    }
    Logger logger;
    logger.sendZmqValue(header, loopPos, values);
}

/**
 * @brief Global wrapper to log errors.
 *
 * @param fctname Name of the function in which is the error.
 * @return Stream Logger::m_logStream.message;
 *
 * @note Use it as:
 * \code{.cpp}
 *      Logger::error(_ME_) << "This is an error;
 * \endcode
 * A prepocessor macro `_ME_` is used for `__PRETTY_FUNCTION__`.
 */
inline Logger error(std::string fctname) { return Logger(LogType::Error, "in " + fctname); }

/**
 * @brief Global function to post an error code on the RFM.
 */
void postError(unsigned int errornr);

/**
 * @brief Global function to express an error code in a verbose way.
 */
std::string errorMessage(unsigned int errornr);

}

#endif // LOGGER_H
