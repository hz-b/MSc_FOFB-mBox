#include "modules/zmq/messenger.h"
#include "modules/zmq/logger.h"

#include <algorithm>


/** Just so that it don't get sent by  error */
#define STOP_MESSAGE "STOP-NOW"
#define STOP_SOCKET "406812310648"

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

    m_map.update("AMPLITUDE-REF-10", static_cast<double>(1));
    m_map.update("PHASE-REF-10", static_cast<double>(0));
    m_editableKeys.push_back("AMPLITUDE-REF-10");
    m_editableKeys.push_back("PHASE-REF-10");

    m_socket = new zmq_ext::socket_t(context, ZMQ_ROUTER);
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
        std::vector<zmq::message_t> request(3);
        m_socket->recv(&request[0]);
        m_socket->recv(&request[1]); // empty delimiter => drop
        while (m_socket->recv(&(request.back()), ZMQ_RCVMORE)) {
            request.push_back(zmq::message_t());
        }
        std::string identity((char*)request[0].data(), request[0].size());
        std::string message((char*)request[2].data(), request[2].size());
        std::transform(message.begin(), message.end(), message.begin(), ::toupper);

        m_socket->send(identity, ZMQ_SNDMORE);
        m_socket->send(std::string(""), ZMQ_SNDMORE); // empty delimiter

        if (message == "KEYLIST" || message == "HELP") {
            this->serveHelp();
        } else if (message == STOP_MESSAGE && identity == STOP_SOCKET) {
            m_serve = false;
            std::string s = "ACK";
            m_socket->send(s);
        } else if (!message.compare(0, prefixSet.length(), prefixSet) && request.size() > 3) {
            std::string key = message.substr(prefixGet.length(), message.npos);
            this->serveSet(key, request[4]);
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
    s = "Use: HELP, KEYLIST, SET <KEY> <VALUE>, GET <KEY>\n\n";
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
void Messenger::Messenger::serveSet(const std::string& key, const zmq::message_t& request)
{
    bool editable = (std::find(m_editableKeys.begin(), m_editableKeys.end(), key) != m_editableKeys.end());
    if (m_map.has(key) && editable) {
        m_map.update(key, (unsigned char*) request.data(), request.size());
        std::string s = "ACK";
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
    Logger::Logger() << "Starting server thread...";
    std::string addr = "tcp://*:" + std::to_string(m_port);
    m_socket->bind(addr.c_str());
    m_serve = true;
    m_serverThread = std::thread(&Messenger::Messenger::servingLoop, this);
    Logger::Logger() << "Server started.";

}

void Messenger::Messenger::stopServing()
{
    Logger::Logger() << "Stopping server thread...";
    zmq::context_t tmp_context(1);
    zmq_ext::socket_t socket_stop(tmp_context, ZMQ_REQ);
    std::string address = "tcp://localhost:" + std::to_string(m_port);
    std::string id = STOP_SOCKET;
    socket_stop.setsockopt(ZMQ_IDENTITY, id.c_str(), id.length());

    socket_stop.connect(address.c_str());
    socket_stop.send(std::string(STOP_MESSAGE));
    zmq::message_t request;
    socket_stop.recv(&request);

    m_serverThread.join();
    Logger::Logger() << "Server stopped, thread joined.";
}

void Messenger::Messenger::setPort(const int port)
{
    m_port = port;
}

int Messenger::Messenger::port() const
{
    return m_port;
}
