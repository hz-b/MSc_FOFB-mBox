#include  "logger/logger.h"

#include <thread>

Logger::Logger::Logger(zmq::context_t &context)
    : m_rfmHelper(NULL, NULL)
{
    m_zmqSocket = new zmq::socket_t(context, zmq::socket_type::pub);
    m_zmqSocket->bind("tcp://*:5555");
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

void Logger::Logger::sendZmq(std::string message)
{
    zmq_ext::send(*m_zmqSocket, message);
}
namespace Logger {
    zmq::context_t context(1);
    Logger logger(context);
}
