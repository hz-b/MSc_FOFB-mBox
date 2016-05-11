#ifndef ZMQEXT_H
#define ZMQEXT_H

#include <rfm2g_api.h>
#include <armadillo>
#include "zmq.hpp"

namespace zmq_ext {

/**
 * @brief Extend the zmq::socket class for more types.
 */
class socket_t : public zmq::socket_t
{
public:
    /**
     * @brief Constructor
     */
    explicit socket_t(zmq::context_t& c, int socket_type);

    /**
     * @brief Send an integer.
     * @param value
     * @param flag
     *
     * @return True if successed
     */
    bool send(const int value, int flags=0);

    /**
     * @brief Send an vector of short.
     * @param value
     * @param flag
     *
     * @return True if successed
     */
    bool send(const std::vector<short> & values, int flags=0);

    /**
     * @brief Send a arma::vec value.
     * @param value
     * @param flag
     *
     * @return True if successed
     */
    bool send(const arma::vec & values, int flags=0);

    /**
     * @brief Send a string.
     * @param value
     * @param flag
     *
     * @return True if successed
     */
    bool send(const std::string & value, int flags=0);
};
}

#endif
