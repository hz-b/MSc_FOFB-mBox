#include "error.h"

Error::Error::Error(const unsigned int type) {
    m_code = type;
    switch (type) {
    case NoError:
        m_type = "";
        m_message = "No Error";
        break;
    case ADC:
        m_type = "MDI Error";
        m_message = "ADC Timeout";
        break;
    case ADCReset:
        m_type = "MDI error";
        m_message = "MDI was restaret";
        break;
    case DAC:
        m_type = "IOC error";
        m_message = "DAC Problem";
        break;
    case CM100:
        m_type = "FOFB error";
        m_message = "To much to correct";
        break;
    case RMS:
        m_type = "FOFB error";
        m_message = "Bad RMS";
        break;
    case NoBeam:
        m_type = "Error";
        m_message = "No Current";
        break;
    default:
        m_type = "Error";
        m_message = "Unknown Problem";
        break;
    }
}
