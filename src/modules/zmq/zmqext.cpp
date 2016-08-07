/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "modules/zmq/zmqext.h"

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
