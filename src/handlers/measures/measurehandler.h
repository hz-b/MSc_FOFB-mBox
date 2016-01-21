#ifndef MEASUREHANDLER_H
#define MEASUREHANDLER_H

#include "handlers/handler.h"

class MeasureHandler : public Handler
{
public:
    explicit MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr);
    virtual int make();

private:
    void setProcessor(arma::mat SmatX, arma::mat SmatY,
                      double IvecX, double IvecY,
                      double Frequency, 
                      double P, double I, double D,
                      arma::vec CMx, arma::vec CMy, bool weightedCorr);

    int m_sample;
    int m_maxSample;
    
    double m_CM;
    double m_amp;
    double m_f;
    double m_fmax;
    int m_CMidx;
    int m_nbCM;
    
};

#endif // HANDLER_H
