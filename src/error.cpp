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
