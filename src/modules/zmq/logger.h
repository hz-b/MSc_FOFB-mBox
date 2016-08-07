/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include "define.h"
#include "modules/zmq/zmqext.h"
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
 * @brief Logging Namespace.
 *
 * The subscribers can subscribe to
 *  * FOFB-ADC-DATA, FOFB-BPM-DATA, FOFB-CM-DATA
 *  * LOG
 *  * ERROR
 *
 * The 3 first (type values) are composed of:
 *  0. the header (FOFB-XXX-DATA)
 *  1. the loop position (int between 0 and 512)
 *  2. the type of data ('short' if ADC, else 'double')
 *  3. Array of data (x-axis if BPM or CM)
 *  4. Array of data (only if BPM or CM: y-axis)
 *
 * LOG and ERROR are composed of:
 *  0. the header ('LOG' or 'ERROR')
 *  1. the time of emission
 *  2. The message
 *  3. Sometimes a secondary message (in ERROR: function where it happened)
 *
 * To log an info:
 * \code{.cpp}
 *      Logger::Logger() << "This is a log line";
 * \endcode
 *
 * To log an error:
 * \code{.cpp}
 *      Logger::error(_ME_) << "This is an error description";
 * \endcode
 *
 * To output data (here CM):
 * \code{.cpp}
 *      Logger::values(LogValue::CM, m_dma->status()->loopPos, std::vector<arma::vec>({CMx, CMy}));
 * \endcode
 *
 */
namespace Logger {

struct log_stream_t {
    log_stream_t(LogType type, std::string s_other = "") : header(type), other(s_other) {}
    LogType header;
    std::ostringstream message;
    std:: string other;
};

/**
 * @brief Class to deal with logs and errors
 *
 * Can be called directly as
 * ~~~~{.cpp}
 * Logger::Logger() << "This is a log line
 * ~~~~
 * The log is output at destruction of the object.
 *
 * Relies on some static variables (m_debug, m_port, m_socket, m_driver).
 *
 * @see See \ref #Logger to see  more detailled information.
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

    void sendMessage(const std::string &message, const std::string &errorType=" ");
    void sendZmq(const std::string& header, const std::string& message, const std::string& other);

    template <typename T>
    void sendZmqValue(const std::string& header, const int loopPos, const std::vector<T>& values)
    {
        if (values.size() == 0) {
            "ERROR -- Tried to send empty values, return";
            return;
        }

        try {
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

        } catch (zmq::error_t &e) {
            if (e.num() != EINTR) {
                throw;
            }
        }
    }

    /**
     * @brief Send message to the RFM.
     * @param message message
     * @param error type of error (= Status)
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
 * Use it as:
 * \code{.cpp}
 *      Logger::error(_ME_) << "This is an error;
 * \endcode
 * A prepocessor macro `_ME_` is used for `__PRETTY_FUNCTION__`.
 */
inline Logger error(std::string fctname) { return Logger(LogType::Error, "in " + fctname); }

/**
 * @brief Global function to post an error code on the RFM.
 */
void postError(const unsigned int errornr);

/**
 * @brief Global function to express an error code in a verbose way.
 */
std::string errorMessage(unsigned int errornr);

}

#endif // LOGGER_H
