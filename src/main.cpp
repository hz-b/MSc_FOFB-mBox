#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>
#include <string>

#include "define.h"
#include "mbox.h"
#include "logger/logger.h"
#include "logger/messenger.h"

bool READONLY; /**< Global variable to authorize or not to write to the RFM */

// Must be started first to be deleted last.
zmq::context_t context(1); /**< ZMQ Context */
/** Global socket used for the logging */
zmq_ext::socket_t logSocket(context, ZMQ_PUB /*zmq::socket_type::pub*/);

/**
 * @brief Static mBox object.
 * @note Must be static so that exit() do a proper deletion.
 */
static mBox mbox;

namespace Messenger {
    /**
     * @brief Global Messenger object.
     */
    Messenger messenger(context);
}

/**
 * @brief Function called on CTRL+C.
 *
 * Exit the program with output 0.
 */
void SIGINT_handler(int signum)
{
    std::cout << "Quit mBox...\n";
    exit(0);
}

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    mbox.parseArgs(argc, argv);

    Logger::setSocket(&logSocket);
    // Wait to be sure that the socket is configured
    std::this_thread::sleep_for(std::chrono::seconds(1));

    mbox.init(DEVICE_NAME, WEIGHTED_CORR);

    signal(SIGINT, SIGINT_handler);
    mbox.startLoop();

    return 0;
}

