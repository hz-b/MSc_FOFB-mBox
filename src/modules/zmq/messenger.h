#ifndef MESSENGER_H
#define MESSENGER_H

#include <armadillo>

#include <string>
#include <thread>
#include <vector>

#include "modules/zmq/extendedmap.h"
#include "modules/zmq/zmqext.h"


namespace Messenger {

/**
 * @class Messenger
 * @brief ZMQ server to exchange parameters with other applications.
 */
class Messenger
{
public:

    /**
     * @brief Constructor. Initialize the socket.
     * @param context
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
    template <typename T>
    void updateMap(const std::string& key, const T& value) {
        m_map.update(key, value);
    }

    /**
     * @brief Shortcut function to get m_map[key]
     */
    void get(const std::string& key, arma::vec& value) const {
        value = m_map.getAsVec(key);
    }

    /**
     * @brief Shortcut function to get m_map[key]
     */
    void get(const std::string& key, double& value) const {
        value = m_map.getAsDouble(key);
    }

private:
    /**
     * @brief Function containing the REQ/REP loop.
     */
    void servingLoop();

    /**
     * @brief Process a `SET` request.
     *
     * @param key Key of the value to set.
     * @param request Contains the value
     */
    void serveSet(const std::string& key, const zmq::message_t& request);

    /**
     * @brief Process a `GET` request.
     *
     * @param key Key of the value to provide
     */
    void serveGet(const std::string& key);

    /**
     * @brief Send the help text.
     */
    void serveHelp();

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
    ExtendedMap m_map;

    /**
     * @brief Return all the keys that can be edited throught the server.
     */
    std::vector<std::string> m_editableKeys;

    /**
     * @brief Server socket
     */
    zmq_ext::socket_t *m_socket;
};

/**
 * @brief Messenger object to used in the global shortcut functions
 */
extern Messenger messenger;

/**
 * @brief Global shortcut to Messenger::updateMap method.
 *
 * @param key Key to update
 * @param value Value to set
 */
template <typename T>
void updateMap(const std::string& key, const T& value) {
    messenger.updateMap(key, value);
}

/**
 * @brief Global shortcut to Messenger::get method.
 *
 * @param[in] key Key of the value to get
 * @param[out] value Value to return
 */
template <typename T>
void get(const std::string& key, T& value) {
    messenger.get(key, value);
}

}
#endif
