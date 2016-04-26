#ifndef ADC_H
#define ADC_H

#include "define.h"
#include <vector>

class RFMDriver;
class DMA;

/**
 * This class is used to read the data (= BPM values) to the RFM.
 */
class ADC
{
public:
    /**
     * @brief Constructor
     */
    explicit ADC(RFMDriver *driver, DMA *dma);

    /**
     * @brief Destructor
     */
    ~ADC();

    /**
     * @brief Initialize the ADC.
     * @return 1 if error, 0 if success
     */
    int init();

    /**
     * @brief Stop the ADC.
     *
     * This must be called when quitting the program, would it be a crash or a normal exit,
     * @return 1 if error, 0 if success
     */
    int stop();

    /**
     * @brief Read the RFM
     *
     * First wait for an interruption from the RFM, then read the RFM into `m_buffer`.
     */
    int read();

    /**
     * @brief Access to an element of the buffer.
     *
     * @param id Index of the buffer element to return
     * @return Buffer element
     */
    RFM2G_INT16 bufferAt(int id) const { if (id < m_buffer.size()) return m_buffer.at(id); };

    /**
     * @brief Return the full buffer vector.
     *
     * @return The buffer vector
     */
    std::vector<RFM2G_INT16> buffer() const { return m_buffer; };

    double waveIndexXAt(int id) const { return m_waveIndexX.at(id); };
    double waveIndexYAt(int id) const { return m_waveIndexY.at(id); };
    void setWaveIndexX(std::vector<double> vect) { m_waveIndexX = vect; };
    void setWaveIndexY(std::vector<double> vect) { m_waveIndexY = vect; };

private:

    /**
     * @brief Procedure to get the authorization to read the RFM.
     *
     * @param eventInfo Information about the event
     * @return Status of the function call
     */
    RFM2G_STATUS waitForEvent(RFM2GEVENTINFO &eventInfo);

    /**
     * @brief Pointer to the DMA object.
     */
    DMA *m_dma;

    /**
     * @brief Pointer to the RFMDriver object.
     */
    RFMDriver *m_driver;

    /**
     * @brief Vector representing the data buffer
     */
    std::vector<RFM2G_INT16> m_buffer;

    std::vector<double> m_waveIndexX;
    std::vector<double> m_waveIndexY;

    /**
     * @brief RFM node with which to communicate.
     */
    int m_node;
};

#endif // ADC_H
