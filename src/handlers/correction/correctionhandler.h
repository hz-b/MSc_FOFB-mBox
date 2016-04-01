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

    ~CorrectionHandler() {};

private:

    /**
     * @brief Calls processor callProcessorRoutine
     */
    int callProcessorRoutine(const arma::vec& diffX, const arma::vec& diffY,
                                     const bool newInjection,
                                     arma::vec& CMx, arma::vec& CMy,
                                     const int typeCorr);

    int typeCorrection();

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
