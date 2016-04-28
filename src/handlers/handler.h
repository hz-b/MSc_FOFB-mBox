#ifndef HANDLER_H
#define HANDLER_H

#include "define.h"
#include "handlers/structures.h"

#include <armadillo>

class ADC;
class DAC;
class DMA;
class RFMDriver;

namespace numbers {
    /**
     * @brief Coefficient to multiply the ADC values with to get real numbers.
     */
    const double cf = 0.3051758e-3;

    /**
     * @brief Offset to get the 0.
     */
    const double halfDigits   = 1<<23;
}

namespace Correction {
    /**
     * @brief Type of correction
     */
    enum Type {
        None = 0b00,
        Horizontal = 0b01,
        Vertical = 0b10,
        All = Horizontal | Vertical
    };
}

/**
 * @class Handler
 * @brief Abstract class to connect the mBox to a Processor that do the maths
 *        to generate the correction.
 */
class Handler
{
public:
    /**
     * @brief Constructor
     *
     * @param driver A pointer to a RFMDriver class.
     * @param dma A pointer to a DMA class.
     * @param weigthedCorr True if we use a weighted correction. Else False.
     */
    explicit Handler(RFMDriver *driver, DMA *dma, bool weigthedCorr);

    ~Handler();

    /**
     * @brief Do what the handler is designed for (correction, setting values..)
     *
     * It calls
     *      * getNewData() to read the BPM values from the RFM
     *      * a function that do the calculations (in the Processor)
     *      * prepareCorrectionValues()
     *      * writeCorrectors() to write the results on the RFM
     */
    int make();

    /**
     * @brief Initialize the attributes and call setProcessor().
     *
     * This will read the RFM to get the parameters from the cBox and initialize the ADC/DAC.
     */
    void init();

    /**
     * @brief Disaqble te ADC and the DAC.
     */
    void disable();

protected:
    /**
     * @brief Read the data given on the RFM.
     *
     * It first wait for the authorization to read. All parameters are filled by the function.
     * @param diffX Values of the BPMs in the x direction (filled by the function)
     * @param diffY Values of the BPMs in the y direction (filled by the function)
     * @param newInjection True if an injection was just sent. Else False.
     *
     * @return error
     */
    int getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection);

    /**
     * @brief Defines the type of correction to perform. Must be implemented in
     * subclasses.
     *
     * @return type of correction
     */
    virtual int typeCorrection() = 0;

     /**
     * @brief Call the processor routine. Must be implemented in subclasses.
     *
     * @param[in] input inputs
     * @param[out] CMx correction to be calculated, x-axis
     * @param[out] CMy correction to be calculated, y-axis
     *
     * @return Error code
     */
    virtual int callProcessorRoutine(const CorrectionInput_t& input,
                                     arma::vec& CMx, arma::vec& CMy) = 0;

    /**
     * @brief Prepare the values to be written by the DAC
     *
     * @param CMx the corrector values for axis x
     * @param CMy the corrector values for axis y
     * @param typeCorr int value in the following set:
     *                  * Correction::None (= `0b00`)
     *                  * Correction::Horizontal (= `0b01`)
     *                  * Correction::Vertical (= `0b10`)
     *                  * Correction::All (= `0b11`)
     * @return DACout, the pointer of values to use in writeCorrectors()
     */
    void prepareCorrectionValues(const arma::vec &CMx, const arma::vec &CMy, int typeCorr);

    /**
     * @brief Write DACout to the RFM
     */
    void writeCorrection();

    /**
     * @brief Get the index of a given index
     * @return Error code.
     */
    int getIdx(const std::vector<double> &ADC_BPMIndex_Pos, double DeviceWaveIndex);

    /**
     * @brief Get the index of a given index
     */
    void initIndexes(const std::vector<double> &ADC_WaveIndexX);

    /**
     * @brief Define the processor and its parameters.
     *
     * This is where a processor should be instanciated.
     */
    virtual void setProcessor(arma::mat SmatX, arma::mat SmatY,
                              double IvecX, double IvecY,
                              double Frequency,
                              double P, double I, double D,
                              arma::vec CMx, arma::vec CMy,
                              bool weightedCorr) = 0;
    ADC *m_adc;
    DAC *m_dac;
    DMA *m_dma;
    RFMDriver *m_driver;
    bool m_weightedCorr;

    int m_idxHBP2D6R,
        m_idxBPMZ6D6R,
        m_idxHBP1D5R,
        m_idxBPMZ3D5R,
        m_idxBPMZ4D5R,
        m_idxBPMZ5D5R,
        m_idxBPMZ6D5R;
    double m_loopDir;
    double m_plane;
    Pair_t<arma::vec> m_scaleDigits;

    Pair_t<arma::vec> m_gain;
    Pair_t<int> m_numBPM;
    Pair_t<int> m_numCM;
    Pair_t<arma::vec> m_BPMoffset;

    RFM2G_UINT32 m_DACout[DAC_BUFFER_SIZE];
};

#endif // HANDLER_H
