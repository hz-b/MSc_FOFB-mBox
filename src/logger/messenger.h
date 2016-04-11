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
    Messenger(zmq::context_t& context);
    ~Messenger();

    void startServing();
    void stopServing();

    void setPort(const int port);
    int port() const;
    template <class T>
    void updateMap(const std::string& key, const T& value) {
        m_map.update(key, value);
    }

private:
    void servingLoop();

    int m_port;
    bool m_serve;
    std::thread m_serverThread;
    Map m_map;
    zmq_ext::socket_t *m_socket;
};

extern Messenger messenger;

template <class T>
void updateMap(const std::string& key, const T& value) {
    messenger.updateMap(key, value);
}

}
#endif
