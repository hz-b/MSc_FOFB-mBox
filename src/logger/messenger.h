#ifndef MESSENGER_H
#define MESSENGER_H

#include <armadillo>

#include <string>
#include <thread>
#include <vector>

#include "logger/map.h"
#include "logger/zmqext.h"

namespace Messenger {

class Messenger
{
public:
    /**
     * @brief Constructor. Initialize the socket.
     */
    Messenger(zmq::context_t& context);

    /**
     * @brief Destructor. Will verify that the server is correctly stopped.
     */
    ~Messenger();

    /**
     * @brief Start serving.
     *
     * The socket is bounded with given port and the servingLoop loop function
     * is called in a separate thread m_serverThread.
     */
    void startServing();

    /**
     * @brief Stop serving.
     *
     * If currently serving, a REQ client is created and sends the terminating
     * commamd. Then the separate thread m_thread is joined.
     */
    void stopServing();

    /**
     * @brief Set the port of the server.
     * @param port integer of the port. No validation is done here.
     */
    void setPort(const int port);

    /**
     * @brief Get the port of the server.
     * @return port of the server.
     */
    int port() const;

    /**
     * @brief Shortcut function to update m_map
     */
    template <class T>
    void updateMap(const std::string& key, const T& value) {
        m_map.update(key, value);
    }

private:
    /**
     * @brief Function containing the REQ/REP loop.
     */
    void servingLoop();

    /**
     * @brief Port used by the server
     */
    int m_port;

    /**
     * @brief Whether the server currently serves or not.
     */
    bool m_serve;

    /**
     * @brief Thread for the serving loop
     */
    std::thread m_serverThread;

    /**
     * @brief Map / Dictionary containing the information that can be queried.
     */
    Map m_map;

    /**
     * @brief Server socket
     */
    zmq_ext::socket_t *m_socket;
};

extern Messenger messenger;

template <class T>
void updateMap(const std::string& key, const T& value) {
    messenger.updateMap(key, value);
}

}
#endif
