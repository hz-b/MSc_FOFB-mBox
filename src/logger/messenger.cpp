#include "logger/messenger.h"
#include "logger/logger.h"

// Just so that it don't get sent by  error
#define STOPPING_MESSAGE "STOP-ME-406812310648"

Messenger::Messenger::Messenger(zmq::context_t& context)
    : m_serve(false)
{
    m_socket = new zmq_ext::socket_t(context, ZMQ_REP);
    m_socket->bind("tcp://*:3334");
    this->startServing();
}

Messenger::Messenger::~Messenger()
{
    this->stopServing();
    delete m_socket;
}

void Messenger::Messenger::servingLoop()
{
    while (m_serve) {
        zmq::message_t request;
        m_socket->recv(&request);
        std::string message((char*)request.data(), request.size());

        if (message == "KEYLIST") {
            std::string s = m_map.keyList();
            m_socket->send(s);
        } else if (message == STOPPING_MESSAGE) {
            m_serve = false;
            std::string s = "ACK";
            m_socket->send(s);
        } else if (m_map.has(message)) {
            int size = m_map.get_sizeof(message);
            zmq::message_t msg(size);
            memcpy(msg.data(), m_map.get_raw(message), size);
            m_socket->zmq::socket_t::send(msg);
        } else {
            std::cout << "error: unknown key/message " << message<<'\n';
            std::string s = "ERROR";
            m_socket->send(s);
        }
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
    zmq_ext::socket_t socket_stop(tmp_context, ZMQ_REQ);
    socket_stop.connect("tcp://localhost:3334");
    socket_stop.send(std::string(STOPPING_MESSAGE));
    zmq::message_t request;
    socket_stop.recv(&request);

    m_serverThread.join();
}

