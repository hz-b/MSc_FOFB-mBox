#ifndef MEASUREHANDLER_H
#define MEASUREHANDLER_H

#include "handlers/handler.h"

#include <Python.h>


/**
 * This class is designed to call a Python functions each time `make()` is called.
 * The `inputFile`, MUST follow the following pattern:
 * 
 * ~~~ Python Code ~~~
 * import numpy as np
 * 
 * def corr_value(BPMx, BPMy, CMx_nb, CMy_nb):
 *     CMx = np.array(CMx)
 *     CMy = nb.array(CMy)
 * 
 *     ...do something here...
 *     
 *     return CMx, CMy
 * 
 * ~~~ End Code ~~~
 */
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
    /**
     * Calls the python function defined in the constructor.
     */
    int callPythonFunction(const arma::vec &BPMx, const arma::vec &BPMy,
                           arma::vec &CMx, arma::vec &CMy);

    PyObject *m_pFunc;
    std::string m_inputFile;
    std::string m_inputPath;
    std::string m_inputModule;
    std::string m_functionName;
};

#endif // HANDLER_H
