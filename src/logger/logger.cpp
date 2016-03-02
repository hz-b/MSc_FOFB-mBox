#include  "logger/logger.h"

#include <ctime>

Logger::Logger::Logger(zmq::context_t &context)
    : m_rfmHelper(NULL, NULL)
{
    m_zmqSocket = new zmq_ext::socket_t(context, ZMQ_PUB /*zmq::socket_type::pub*/);
    m_zmqSocket->bind("tcp://*:3333");
}

Logger::Logger::~Logger()
{
    delete m_zmqSocket;
}

void Logger::Logger::record(std::string message)
{
    std::cout << message << std::endl;;
}

void Logger::Logger::sendMessage(std::string message, std::string error)
{
    if (READONLY) {
        std::cout << "Message: " << message;
        if (!error.empty())
            std::cout << " Error: " << error;
        std::cout << std::endl;
    } else {
        m_rfmHelper.sendMessage(message.c_str(), error.c_str());
    }
}

void Logger::Logger::sendZmq(const std::string& header, const std::string& message)
{
    time_t rawtime = time(NULL);
    struct tm* timeinfo = localtime(&rawtime);
    char timeBuf[80];
    strftime(timeBuf, sizeof(timeBuf), "%F %T", timeinfo);

    std::string formatedMessage = "[" + header + "] " + std::string(timeBuf) + " -- " + message;
    m_zmqSocket->send(formatedMessage);
}
