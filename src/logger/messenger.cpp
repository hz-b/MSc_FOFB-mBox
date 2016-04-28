#include "logger/messenger.h"
#include "logger/logger.h"

#include <algorithm>


/** Just so that it don't get sent by  error */
#define STOPPING_MESSAGE "STOP-ME-406812310648"

Messenger::Messenger::Messenger(zmq::context_t& context)
    : m_serve(false)
{
    m_map.update("AMPLITUDES-X-10", arma::vec());
    m_map.update("PHASES-X-10", arma::vec());
    m_editableKeys.push_back("AMPLITUDES-X-10");
    m_editableKeys.push_back("PHASES-X-10");

    m_map.update("AMPLITUDES-Y-10", arma::vec());
    m_map.update("PHASES-Y-10", arma::vec());
    m_editableKeys.push_back("AMPLITUDES-Y-10");
    m_editableKeys.push_back("PHASES-Y-10");

    m_socket = new zmq_ext::socket_t(context, ZMQ_REP);
    m_port = 3334;
}

Messenger::Messenger::~Messenger()
{
    if (m_serve) {
        this->stopServing();
    }
    delete m_socket;
}

void Messenger::Messenger::servingLoop()
{
    std::string prefixSet = "SET ";
    std::string prefixGet = "GET ";
    while (m_serve) {
        zmq::message_t request;
        m_socket->recv(&request);
        std::string message((char*)request.data(), request.size());
        std::transform(message.begin(), message.end(), message.begin(), ::toupper);
        if (message == "KEYLIST" || message == "HELP") {
            this->serveHelp();
        } else if (message == STOPPING_MESSAGE) {
            m_serve = false;
            std::string s = "ACK";
            m_socket->send(s);
        } else if (!message.compare(0, prefixSet.length(), prefixSet)) {
            std::string key = message.substr(prefixGet.length(), message.npos);
            this->serveSet(key);
        } else if (!message.compare(0, prefixGet.length(), prefixGet)) {
            std::string key =  message.substr(prefixGet.length(), message.npos);
            this->serveGet(key);
        } else {
            std::cout << "error: unknown key/message " << message<<'\n';
            std::string s = "KEY ERROR";
            m_socket->send(s);
        }
    }
}

void Messenger::Messenger::serveHelp()
{
    std::string s;
    s = "Use: HELP, KEYLIST, SET <KEY>, GET <KEY>\n\n";
    s += "AVAILABLE KEYS TO GET\n"
         "=====================\n"
         + m_map.keyList() + '\n';
    s += "AVAILABLE KEYS TO SET\n"
         "=====================\n";
    for (const std::string& key : m_editableKeys) {
        s += key + '\n';
    }
    m_socket->send(s);
}
void Messenger::Messenger::serveSet(const std::string& key)
{
    bool editable = (std::find(m_editableKeys.begin(), m_editableKeys.end(), key) != m_editableKeys.end());
    if (m_map.has(key) && editable) {
        std::string s = "GO";
        m_socket->send(s);
        zmq::message_t request;
        m_socket->recv(&request);
        m_map.update(key, (unsigned char*) request.data(), request.size());
        s = "ACK";
        m_socket->send(s);
    } else {
        std::string s = "KEY ERROR";
        m_socket->send(s);
    }
}

void Messenger::Messenger::serveGet(const std::string& key)
{
    if (m_map.has(key)) {
        int size = m_map.get_sizeof(key);
        zmq::message_t msg(size);
        memcpy(msg.data(), m_map.get_raw(key), size);
        m_socket->zmq::socket_t::send(msg);
    } else {
        std::string s = "KEY ERROR";
        m_socket->send(s);
    }
}

void Messenger::Messenger::startServing()
{
    Logger::Logger() << "Starting server thread.";
    std::string addr = "tcp://*:" + std::to_string(m_port);
    m_socket->bind(addr.c_str());
    m_serve = true;
    m_serverThread = std::thread(&Messenger::Messenger::servingLoop, this);
}

void Messenger::Messenger::stopServing()
{
    Logger::Logger() << "Stopping server thread.";
    zmq::context_t tmp_context(1);
    zmq_ext::socket_t socket_stop(tmp_context, ZMQ_REQ);
    socket_stop.connect("tcp://localhost:3334");
    socket_stop.send(std::string(STOPPING_MESSAGE));
    zmq::message_t request;
    socket_stop.recv(&request);

    m_serverThread.join();
}

void Messenger::Messenger::setPort(const int port)
{
    m_port = port;
}

int Messenger::Messenger::port() const
{
    return m_port;
}
