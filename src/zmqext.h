#ifndef ZMQEXT_HP
#define ZMQEXT_H

#include <armadillo>
#include "zmq.hpp"
//#include "zhelpers.hpp"

// RFM2G_INT16    ADC_Buffer[ADC_BUFFER_SIZE];

bool send(zmq::socket_t & socket, RFM2G_INT16* value , int size, int flags =0)
{
    int len = sizeof(RFM2G_INT16)*size;
    zmq::message_t msg(len);
    memcpy(msg.data(), value, len);
    return socket.send(msg, flags);
}


bool send(zmq::socket_t & socket, int value, int flags =0)
{
    zmq::message_t msg(sizeof(int));
    memcpy(msg.data(), &value, sizeof(int));
    return socket.send(msg, flags);
}


bool send(zmq::socket_t & socket, std::vector<short> & values, int flags =0)
{
    int len=sizeof(short)*values.size();
    zmq::message_t msg(len);
    memcpy(msg.data(), values.data(), len);
    return socket.send(msg, flags);
}


bool send(zmq::socket_t & socket, arma::vec & values, int flags =0)
{
    int len=sizeof(double)*values.n_elem;
    zmq::message_t msg(len);
    memcpy(msg.data(), values.memptr(), len);
    return socket.send(msg, flags);
}

#endif