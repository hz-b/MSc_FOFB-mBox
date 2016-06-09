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
