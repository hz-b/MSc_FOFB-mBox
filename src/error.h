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

#ifndef ERROR_H
#define ERROR_H

#include <string>

/**
 * @brief Error Namespace
 */
namespace Error {
    /**
     * @brief Error numbers
     */
    enum ErrorCode : unsigned int {
        NoError     = 0, /**< @brief No error. */
        ADC         = 1, /**< @brief Error in ADC. */
        DAC         = 2, /**< @brief Error in DAC. */
        CM100       = 4, /**< @brief Correction output to great. */
        NoBeam      = 5, /**< @brief No beam (BPM input to low). */
        RMS         = 6, /**< @brief RMS error (Correction doesn't improve). */
        ADCReset    = 8, /**< @brief ADC was reset */
        Unkonwn     = 7  /**< @brief Unknown Error. */
    };

    /**
     * @brief Class for errors
     */
    class Error
    {
    public:
        /**
         * @brief Constructor
         * @param type Error code
         */
        explicit Error(const unsigned int type);

        /**
         * @brief Getter for m_type
         * @return Type of error
         */
        std::string type() { return m_type; }

        /**
         * @brief Getter for m_message
         * @return Error Message
         */
        std::string message() { return m_message; }

        /**
         * @brief Getter for m_code
         * @return Error code, see Error::ErrorCode
         */
        unsigned int code() { return m_code; }

    private:
        std::string m_type; /**< @brief Type of error */
        std::string m_message; /**< @brief Error message */
        unsigned int m_code; /**< @brief Error code */
    };
}

#endif // ERROR_H
