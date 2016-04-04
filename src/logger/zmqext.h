#ifndef ZMQEXT_H
#define ZMQEXT_H

#include <rfm2g_api.h>
#include <armadillo>
#include "zmq.hpp"


// template<T>
// bool send(zmq::socket_t & socket, T value, int flags =0) {
//     return send(socket, value, flags);
// }

namespace zmq_ext {

    class socket_t : public zmq::socket_t
    {
    public:
        explicit socket_t(zmq::context_t& c, int socket_type);
        bool send(const RFM2G_INT16* value , int size, int flags=0);
        bool send(const int value, int flags=0);
        bool send(const std::vector<short> & values, int flags=0);
        bool send(const arma::vec & values, int flags=0);
        bool send(const std::string & value, int flags=0);
    };
}

#endif
