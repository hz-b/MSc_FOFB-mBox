#ifndef MEASUREHANDLER_H
#define MEASUREHANDLER_H

#include "handlers/handler.h"

#include <Python.h>

class MeasureHandler : public Handler
{
public:
    explicit MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr,
                            std::string inputFile);
    ~MeasureHandler();
    virtual int make();

private:
    void setProcessor(arma::mat SmatX, arma::mat SmatY,
                      double IvecX, double IvecY,
                      double Frequency, 
                      double P, double I, double D,
                      arma::vec CMx, arma::vec CMy, bool weightedCorr);
    void setModule();
    int initPython();
    int callPythonFunction(const arma::vec &BPMx, const arma::vec &BPMy,
                           arma::vec &CMx, arma::vec &CMy);

    int m_sample;
    int m_maxSample;
    
    double m_CM;
    double m_amp;
    double m_f;
    double m_fmax;
    int m_CMidx;
    int m_nbCM;
    
    int m_timeValue;
    PyObject *m_pFunc;
    std::string m_inputFile;
    std::string m_inputPath;
    std::string m_inputModule;
    std::string m_functionName;
};

#endif // HANDLER_H
