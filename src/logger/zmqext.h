#ifndef ZMQEXT_H
#define ZMQEXT_H

#include <rfm2g_api.h>
#include <armadillo>
#include "zmq.hpp"
//#include "zhelpers.hpp"

// RFM2G_INT16    ADC_Buffer[ADC_BUFFER_SIZE];

// template<T>
// bool send(zmq::socket_t & socket, T value, int flags =0) {
//     return send(socket, value, flags);
// }

namespace zmq_ext {
    bool send(zmq::socket_t & socket, RFM2G_INT16* value , int size, int flags =0);
    bool send(zmq::socket_t & socket, int value, int flags =0);
    bool send(zmq::socket_t & socket, std::vector<short> & values, int flags =0);
    bool send(zmq::socket_t & socket, arma::vec & values, int flags =0);
    bool send(zmq::socket_t & socket, std::string & value, int flags =0);
}

#endif
