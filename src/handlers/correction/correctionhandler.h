#ifndef CORRECTIONHANDLER_H
#define CORRECTIONHANDLER_H

#include "handlers/handler.h"
#include "handlers/correction/correctionprocessor.h"


class CorrectionHandler : public Handler
{
public:
    explicit CorrectionHandler(RFMDriver *driver, DMA *dma, bool weightedCorr);
    virtual int make();

private:
    void setProcessor(arma::mat SmatX, arma::mat SmatY,
                      double IvecX, double IvecY,
                      double Frequency, 
                      double P, double I, double D,
                      arma::vec CMx, arma::vec CMy, 
                      bool weightedCorr);

    CorrectionProcessor m_correctionProcessor;

};

#endif // HANDLER_H
