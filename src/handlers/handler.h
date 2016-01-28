#ifndef HANDLER_H
#define HANDLER_H

#include "define.h"

#include <armadillo>

class ADC;
class DAC;
class DMA;
class RFMDriver;

namespace numbers {
    const double cf = 0.3051758e-3;
    const double halfDigits   = 1<<23;
}

class Handler
{
public:
    /**
     * @brief Constructor
     */
    explicit Handler(RFMDriver *driver, DMA *dma, bool weigthedCorr);
    ~Handler();
    
    /**
     * @brief Do what the handler is designed for (correction, setting values..)
     * This should call `writeCorrectors()` to write the results on the RFM
     */
    virtual int make() = 0;
    
    /**
     * @brief Initialize the parameters and call `setProcessor()`.
     * This will read the RFM to get the parameters from the cBox
     */
    void init();
    
    int status() { return m_status; }
    
protected:
    /**
     * @brief Read the data from the cBox.
     * It first wait for the authorization to read.
     */
    void getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection);
    
    /**
     * @brief Write DACout to the RFM
     */
    void writeCorrectors(RFM2G_UINT32* DACout);
    
    /**
     * @brief Get the index of a given index
     * @return
     */
    int getIdx(char numBPMs, double *ADC_BPMIndex_Pos, double DeviceWaveIndex);
    
    /**
     * @brief Get the index of a given index
     * @return
     */
    void initIndexes(double *ADC_WaveIndexX);
    
    /**
     * @brief Define the processor and its parameters.
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
    arma::vec m_scaleDigitsX, m_scaleDigitsY;

    arma::vec m_gainX, m_gainY;
    int m_numBPMx, m_numBPMy;
    int m_numCMx, m_numCMy;
    arma::vec m_BPMoffsetX, m_BPMoffsetY;
    
    int m_status;
};

#endif // HANDLER_H
