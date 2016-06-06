#include  "modules/zmq/logger.h"

#include <ctime>

bool Logger::Logger::m_debug = false;
zmq_ext::socket_t* Logger::Logger::m_zmqSocket = NULL;
RFMDriver* Logger::Logger::m_driver = NULL;
int Logger::Logger::m_port = 3333;

Logger::Logger::Logger(LogType type, std::string other)
 : m_logStream( new log_stream_t(type, other) )
 {}

Logger::Logger::~Logger()
{
    if (!m_logStream->message.str().empty())
    {
        this->parseAndSend();
    }
    delete m_logStream;
}

void Logger::Logger::parseAndSend()
{
    std::string header;
    switch (m_logStream->header) {
    case LogType::Log:
        header = "LOG";
        if (m_debug) {
            std::clog << '[' << header << "] "
                      << m_logStream->message.str();
            if (!m_logStream->other.empty())
                std::clog << '\t' << m_logStream->other;
            std::clog << '\n';
        }
        break;
    case LogType::Error:
        header = "ERROR";
        std::cerr << "\x1b[1;31m[" << header << "]\x1b[0m "
                  << m_logStream->message.str() << "\t\x1b[31m[" << m_logStream->other << "]\x1b[0m\n";
    }
    if (m_zmqSocket != NULL) {
       this->sendZmq(header, m_logStream->message.str(), m_logStream->other);
    }
}

void Logger::Logger::setSocket(zmq_ext::socket_t* socket)
{
    m_zmqSocket = socket;
    std::string addr = "tcp://*:" + std::to_string(m_port);
    m_zmqSocket->bind(addr.c_str());
}

void Logger::Logger::setPort(const int port)
{
    m_port = port;
}

int Logger::Logger::port() const
{
    return m_port;
}

void Logger::Logger::sendMessage(const std::string& message,
                                 const std::string& errorType)
{
    // error are already shown by Logger::error()
    if (errorType == " ") {
        std::cout << "Status: " << message << '\n';
    }

    if (!READONLY) {
        this->sendRFM(message, errorType);
    }
}

void Logger::Logger::sendRFM(const std::string& message, const std::string& error)
{
    unsigned long pos = MESSAGE_MEMPOS;
    //cout << "Send To Pos: " << pos << endl;
    struct t_header {
        unsigned short namesize;
        unsigned short sizey;
        unsigned short sizex;
        unsigned short type;
    } header;
    int thesize = 2 + sizeof(header)+ 6 + message.length() +
            sizeof(header)+ 5 + error.length() ;
    unsigned char * mymem = (unsigned char *) malloc(thesize);
    unsigned long structpos = 0;
    mymem[0]=2;  mymem[1] = 0; structpos += 2;// number of Elements (message, error)
    header.namesize = 6;
    header.sizex = message.length();
    header.sizey = 1;
    header.type = 2;
    memcpy(mymem+structpos, &header, sizeof(header));
    structpos += sizeof(header);
    memcpy(mymem+structpos, "status",6);
    structpos += 6;
    memcpy(mymem+structpos, message.c_str(), message.length());
    structpos += message.length();

    header.namesize=5;
    header.sizex = error.length();
    header.sizey = 1;
    header.type = 2;
    memcpy(mymem+structpos,&header,sizeof(header));
    structpos += sizeof(header);
    memcpy(mymem+structpos,"error",5);
    structpos += 5;
    memcpy(mymem+structpos,error.c_str(), error.length());
    structpos += error.length();

    m_driver->write(pos , mymem, thesize);
    //unsigned short l = 2;
    //result = RFM2gWrite( RFM_Handle, pos , &l, 2);
    //cout << "Result" << result << endl;
    free(mymem);
}

void Logger::Logger::sendZmq(const std::string& header, const std::string& message, const std::string& other)
{
    std::time_t rawtime = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&rawtime));
    time = time.substr(0, time.size()-1); // Remove '\n' at the end of the string

    m_zmqSocket->send(header, ZMQ_SNDMORE);
    m_zmqSocket->send(time, ZMQ_SNDMORE);

    if (!other.empty()) {
        Logger::m_zmqSocket->send(message, ZMQ_SNDMORE);
        Logger::m_zmqSocket->send(other);
    } else {
        Logger::m_zmqSocket->send(message);
    }
}

// Global functions
void Logger::setDebug(bool debug)
{
    Logger logger;
    logger.setDebug(debug);
}

void Logger::setSocket(zmq_ext::socket_t* socket)
{
    Logger logger;
    logger.setSocket(socket);
}

void Logger::setPort(const int port)
{
    Logger logger;
    logger.setPort(port);
}

void Logger::postError(unsigned int errornr)
{
    Logger logger;
    if (errornr) {
        std::pair<std::string, std::string> error = errorMessage(errornr);
        logger.sendMessage(error.first, error.second);
    }
}

std::pair<std::string, std::string> Logger::errorMessage(unsigned int errornr)
{
    std::pair<std::string, std::string> message;
    switch (errornr) {
    case 0:
        message = {"", "No Error"};
        break;
    case FOFB_ERROR_ADC:
        message = {"MDI Error", "ADC Timeout"};
        break;
    case FOFB_ERROR_ADCReset:
        message = {"MDI error", "MDI was restaret"};
        break;
    case FOFB_ERROR_DAC:
        message = {"IOC error", "DAC Problem"};
        break;
    case FOFB_ERROR_CM100:
        message = {"FOFB error", "To much to correct"};
        break;
    case FOFB_ERROR_RMS:
        message = {"FOFB error", "Bad RMS"};
        break;
    case FOFB_ERROR_NoBeam:
        message = {"Error", "No Current"};
        break;
    default:
        message = {"Error", "Unknown Problem"};
        break;
    }
    return message;
}
