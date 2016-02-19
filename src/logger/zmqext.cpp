#include "logger/zmqext.h"


bool zmq_ext::send(zmq::socket_t & socket, RFM2G_INT16* value , int size, int flags)
{
    int len = sizeof(RFM2G_INT16)*size;
    zmq::message_t msg(len);
    memcpy(msg.data(), value, len);
    return socket.send(msg, flags);
}


bool zmq_ext::send(zmq::socket_t & socket, int value, int flags)
{
    zmq::message_t msg(sizeof(int));
    memcpy(msg.data(), &value, sizeof(int));
    return socket.send(msg, flags);
}


bool zmq_ext::send(zmq::socket_t & socket, std::vector<short> & values, int flags)
{
    int len=sizeof(short)*values.size();
    zmq::message_t msg(len);
    memcpy(msg.data(), values.data(), len);
    return socket.send(msg, flags);
}


bool zmq_ext::send(zmq::socket_t & socket, arma::vec & values, int flags)
{
    int len=sizeof(double)*values.n_elem;
    zmq::message_t msg(len);
    memcpy(msg.data(), values.memptr(), len);
    return socket.send(msg, flags);
}

bool zmq_ext::send(zmq::socket_t & socket, std::string & value, int flags)
{
    int len=value.size()+1; // +1 need for the NULL character at the end.
    zmq::message_t msg(len);
    memcpy(msg.data(), value.c_str(), len);
    return socket.send(msg, flags);
}
