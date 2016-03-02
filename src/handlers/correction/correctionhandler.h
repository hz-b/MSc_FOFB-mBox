#ifndef CORRECTIONHANDLER_H
#define CORRECTIONHANDLER_H

#include "handlers/handler.h"
#include "handlers/correction/correctionprocessor.h"

/**
 * @class CorrectionHandler
 * @brief Implement the Handler class and delegate the mass to CorrectionProcessor.
 */
class CorrectionHandler : public Handler
{
public:
    /**
     * @brief Constructor
     */
    explicit CorrectionHandler(RFMDriver *driver, DMA *dma, bool weightedCorr);

    /**
     * @brief Reads the data and ask for the processor to calculate the correction.
     */
    virtual int make();

private:
    /**
     * @brief Set the processor: the S matrix, the PID values and other parameters are initialized here.
     */
    void setProcessor(arma::mat SmatX, arma::mat SmatY,
                      double IvecX, double IvecY,
                      double Frequency,
                      double P, double I, double D,
                      arma::vec CMx, arma::vec CMy,
                      bool weightedCorr);

    CorrectionProcessor m_correctionProcessor;

};

#endif // HANDLER_H
