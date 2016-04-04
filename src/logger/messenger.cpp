#include "logger/messenger.h"

Messenger::Messenger::Messenger(zmq::context_t& context)
    : m_serve(false)
{
    m_socket = new zmq::socket_t(context, ZMQ_REP);
    m_socket->bind("tcp://*:3334");
    this->startServing();
}

Messenger::Messenger::~Messenger()
{
    m_socket->close();
    delete m_socket;
    this->stopServing();
}

void Messenger::Messenger::servingLoop()
{
    while (m_serve) {
        zmq::message_t request;
        m_socket->recv(&request);
        std::string key((char*)request.data());
        const void* content;
        int size;
        if (key == "KEYLIST") {
            std::string s = m_map.keyList();
            content = static_cast<const void*>(s.data());
            size = s.size();
        } else if (m_map.has(key)) {
            content = static_cast<const void*>(m_map.get_raw(key));
            size = m_map.get_sizeof(key);
        } else {
            std::string s = "ERROR";
            content = static_cast<const void*>(s.data());
            size = s.size();
        }
        m_socket->send(content, size);
    }
}

void Messenger::Messenger::startServing()
{
    std::cout << "Starting thread caller.\n";
    m_serve = true;
    m_serverThread = std::thread(&Messenger::Messenger::servingLoop, this);
    m_serverThread.detach();
}

void Messenger::Messenger::stopServing()
{
    std::cout << "Stopping thread caller.\n";
    m_serve = false;
    m_serverThread.join();
}

