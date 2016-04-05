#include "logger/messenger.h"
#include "logger/logger.h"

// Just so that it don't get sent by  error
#define STOPPING_MESSAGE "STOP-ME-406812310648"

Messenger::Messenger::Messenger(zmq::context_t& context)
    : m_serve(false)
{
    m_socket = new zmq::socket_t(context, ZMQ_REP);
    m_socket->bind("tcp://*:3334");
    this->startServing();
}

Messenger::Messenger::~Messenger()
{
    this->stopServing();
    m_socket->close();
    delete m_socket;
}

void Messenger::Messenger::servingLoop()
{
    while (m_serve) {
        zmq::message_t request;
        m_socket->recv(&request);
        std::string message((char*)request.data(), request.size());
        const void* content;
        int size;
        if (message == "KEYLIST") {
            std::string s = m_map.keyList();
            content = static_cast<const void*>(s.data());
            size = s.size();
        } else if (message == STOPPING_MESSAGE) {
            m_serve = false;
            std::string s = "ACK";
            content = static_cast<const void*>(s.data());
            size = s.size();
        } else if (m_map.has(message)) {
            content = static_cast<const void*>(m_map.get_raw(message));
            size = m_map.get_sizeof(message);
        } else {
            std::cout << "error\n";
            std::string s = "ERROR";
            content = static_cast<const void*>(s.data());
            size = s.size();
        }
        m_socket->send(content, size);
    }
}

void Messenger::Messenger::startServing()
{
    Logger::log() << "Starting server thread." << Logger::flush;
    m_serve = true;
    m_serverThread = std::thread(&Messenger::Messenger::servingLoop, this);
    //m_serverThread.detach();
}

void Messenger::Messenger::stopServing()
{
    Logger::log() << "Stopping server thread." << Logger::flush;
    zmq::context_t tmp_context(1);
    zmq::socket_t socket_stop(tmp_context, ZMQ_REQ);
    socket_stop.connect("tcp://localhost:3334");
    std::string s = STOPPING_MESSAGE;
    const void* content = static_cast<const void*>(s.data());
    int size = s.size();
    socket_stop.send(content, size);
    zmq::message_t request;
    socket_stop.recv(&request);
    socket_stop.close();
    tmp_context.close();

    m_serverThread.join();
}

