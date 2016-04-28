#include "logger/zmqext.h"

zmq_ext::socket_t::socket_t(zmq::context_t& c, int socket_type)
    : zmq::socket_t(c, socket_type)
{

}

bool zmq_ext::socket_t::send(const int value, int flags)
{
    int len = sizeof(int);
    zmq::message_t msg(len);
    memcpy(msg.data(), &value, len);
    return zmq::socket_t::send(msg, flags);
}

bool zmq_ext::socket_t::send(const std::vector<short> & values, int flags)
{
    int len=sizeof(short)*values.size();
    zmq::message_t msg(len);
    memcpy(msg.data(), values.data(), len);
    return zmq::socket_t::send(msg, flags);
}

bool zmq_ext::socket_t::send(const arma::vec & values, int flags)
{
    int len = sizeof(double)*values.n_elem;
    zmq::message_t msg(len);
    memcpy(msg.data(), values.memptr(), len);
    return zmq::socket_t::send(msg, flags);
}

bool zmq_ext::socket_t::send(const std::string & value, int flags)
{
    int len = value.size();
    zmq::message_t msg(len);
    memcpy(msg.data(), value.c_str(), len);
    return zmq::socket_t::send(msg, flags);
}
