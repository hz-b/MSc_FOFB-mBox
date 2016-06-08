#ifndef CORRECTIONHANDLER_H
#define CORRECTIONHANDLER_H

#include "handlers/handler.h"
#include "handlers/correction/correctionprocessor.h"
#include "handlers/correction/dynamic10hzcorrectionprocessor.h"

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
     * @brief Destructor
     */
    ~CorrectionHandler() {};

private:

    /**
     * @brief Call processor routine that do correction.
     */
    virtual int callProcessorRoutine(const CorrectionInput_t& input,
                                     arma::vec& CMx, arma::vec& CMy);

    /**
     * @brief Return the type of correction wanted.
     */
    virtual int typeCorrection();

    /**
     * @brief Set the processor: the S matrix, the PID values and other
     * parameters are initialized here.
     */
    virtual void setProcessor(arma::mat SmatX, arma::mat SmatY,
                              double IvecX, double IvecY,
                              double Frequency,
                              double P, double I, double D,
                              arma::vec CMx, arma::vec CMy,
                              bool weightedCorr);

    /**
     * @brief Processor, i.e. what does the maths.
     */
    CorrectionProcessor m_correctionProcessor;
    /**
     * @brief Additional processor for the 10Hz harmonic perturbation
     */
    Dynamic10HzCorrectionProcessor m_dyn10HzCorrectionProcessor;

};

#endif // HANDLER_H
