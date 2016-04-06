#include  "logger/logger.h"

#include <ctime>

Logger::Logger::Logger(zmq::context_t& context)
    : m_driver(NULL)
{
    m_zmqSocket = new zmq_ext::socket_t(context, ZMQ_PUB /*zmq::socket_type::pub*/);
    m_zmqSocket->bind("tcp://*:3333");
}

Logger::Logger::~Logger()
{
    delete m_zmqSocket;
}

void Logger::Logger::sendMessage(std::string message, std::string error)
{
    if (!READONLY) {
        this->sendRFM(message, error);
    }
}

void Logger::Logger::sendRFM(std::string message, std::string error)
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
    time_t rawtime = time(NULL);
    struct tm* timeinfo = localtime(&rawtime);
    char timeBuf[80];
    strftime(timeBuf, sizeof(timeBuf), "%F %T", timeinfo);

    std::string time(timeBuf);

    m_zmqSocket->send(header, ZMQ_SNDMORE);
    m_zmqSocket->send(time, ZMQ_SNDMORE);

    if (!other.empty()) {
        m_zmqSocket->send(message, ZMQ_SNDMORE);
        m_zmqSocket->send(other);
    } else {
        m_zmqSocket->send(message);
    }
}

void Logger::Logger::sendZmqValue(const std::string& header, const int loopPos, const arma::vec& valueX, const arma::vec& valueY)
{
    m_zmqSocket->send(header, ZMQ_SNDMORE);
    m_zmqSocket->send(loopPos, ZMQ_SNDMORE);
    m_zmqSocket->send(valueX, ZMQ_SNDMORE);
    m_zmqSocket->send(valueY);

}

// Global functions
void Logger::setDebug(bool debug)
{
    logger.setDebug(debug);
}

std::ostream& Logger::flush(std::ostream& output)
{
    std::string header;
    switch (logger.logStream().header) {
    case LogType::Log:
        header = "LOG";
        if (logger.hasDebug()) {
            std::clog << '[' << header << "] "
                      << logger.logStream().message.str();
            if (!logger.logStream().other.empty())
                std::clog << '\t' << logger.logStream().other;
            std::clog << '\n';
        }
        break;
    case LogType::Error:
        header = "ERROR";
        std::cerr << "\x1b[1;31m[" << header << "]\x1b[0m "
                  << logger.logStream().message.str() << "\t\x1b[31m[" << logger.logStream().other << "]\x1b[0m\n";
    }
    logger.sendZmq(header, logger.logStream().message.str(), logger.logStream().other);
    logger.logStream().header = LogType::None;
    logger.logStream().message.str("");
    logger.logStream().other = "";
}


std::ostringstream& Logger::log(LogType type)
{
    logger.logStream().header = type ;
    return logger.logStream().message;
}

std::ostringstream& Logger::log()
{
    return log(LogType::Log);
}


void Logger::values(LogValue name, const int loopPos, const arma::vec& valueX, const arma::vec& valueY)
{
    std::string header;
    switch (name) {
    case LogValue::BPM:
        header = "FOFB-BPM-DATA";
        break;
    case LogValue::CM:
         header = "FOFB-CM-DATA";
        break;
    default:
        error(_ME_) << "Tried to send values of unexpected type. RETURN";
        return;
    }
    logger.sendZmqValue(header, loopPos, valueX, valueY);
}

std::ostringstream& Logger::error(std::string fctname)
{
    logger.logStream().other = "in " + fctname;
    return log(LogType::Error);
}

void Logger::postError(unsigned int errornr)
{
    if (errornr) {
        logger.sendMessage("FOFB error", errorMessage(errornr));
    }
}

std::string Logger::errorMessage(unsigned int errornr)
{
    std::string message;
    switch (errornr) {
    case 0:
        message = "No Error";
        break;
    case FOFB_ERROR_ADC:
        message = "ADC Timeout";
        break;
    case FOFB_ERROR_DAC:
        message = "DAC Problem";
        break;
    case FOFB_ERROR_CM100:
        message = "To much to correct";
        break;
    case FOFB_ERROR_NoBeam:
        message = "No Current";
        break;
    case FOFB_ERROR_RMS:
        message = "Bad RMS";
        break;
    default:
        message = "Unknown Problem";
        break;
    }
    return message;
}
