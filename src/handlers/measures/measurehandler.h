#ifndef MEASUREHANDLER_H
#define MEASUREHANDLER_H

#include "handlers/handler.h"

#include <Python.h>

/**
 * @brief Name of the function to call
 */
const std::string PYTHON_CORRECTION_FUNCTION = "corr_value";

/**
 * @class MeasureHandler
 * @brief This class is designed to call a Python functions each time `make()` is called.
 *
 * The `inputFile`, MUST follow the following pattern:
 *
 * \code{.py}
 * import numpy as np
 *
 * def init(BPMx_nb, BPMy_nb, CMx_nb, CMy_nb):
 *      global gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb
 *      global ...other global....
 *
 *      gBPMx_nb = BPMx_nb
 *      gBPMy_nb = BPMy_nb
 *      gCMx_nb = CMx_nb
 *      gCMy_nb = CMy_nb
 *
 *      gXXX = .... initialization of other variables ....
 *
 *
 * def corr_value(BPMx, BPMy):
 *     global gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb
 *     CMx = np.array(gCMx_nb)
 *     CMy = nb.array(gCMy_nb)
 *
 *     ...do something here...
 *
 *     return CMx, CMy
 * \endcode
 */
class MeasureHandler : public Handler
{
public:
    /**
     * @brief Constructor
     */
    explicit MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr,
                            std::string inputFile);

    /**
     * @brief Destructor
     */
    ~MeasureHandler();

private:
    /**
     * @brief Set the processor, here Python.
     */
    virtual void setProcessor(arma::mat SmatX, arma::mat SmatY,
                              double IvecX, double IvecY,
                              double Frequency,
                              double P, double I, double D,
                              arma::vec CMx, arma::vec CMy, bool weightedCorr);

    /**
     * @brief Calls processor callProcessorRoutine
     */
    virtual int callProcessorRoutine(const CorrectionInput_t& input,
                                     arma::vec& CMx, arma::vec& CMy);

    /**
     * @brief Return the type of Correction wanted.
     */
    virtual int typeCorrection();

    /**
     * @brief String manipulations to define the Python module to use, based on `m_inputFile`.
     */
    void setModule();

    /**
     * @brief Inititialize Python environnment.
     *
     * It sets the attributes `m_pFunc` and `m_pFunc` and call callPythonInit()
     *
     * @note Should be called only after having set `m_numBPMx/y` and `m_numCMx/y`!
     * @return Error: 0 if success, 1 if failure.
     */
    int initPython();

    /**
     * @brief Call the `init( ..args... )` function from the Python module `m_pModule`.
     */
    int callPythonInit();

    /**
     * @brief Calls the python function defined in the constructor.
     */
    int callPythonFunction(const arma::vec& BPMx, const arma::vec& BPMy,
                           arma::vec& CMx, arma::vec& CMy);

    /**
     * @brief Python object representing the function to be called for the
     * correction.
     */
    PyObject *m_pFunc;

    /**
     * @brief Python object representing the module where m_pFunc can be found.
     */
    PyObject *m_pModule;

    /**
     * @brief Full path of the input file (e.g. `path/to/input_file.py`).
     */
    std::string m_inputFile;

    /**
     * @brief Path of the input file (e.g. `path/to`).
     */
    std::string m_inputPath;

    /**
     * @brief Name of the Python module = file (e.g `input_file.py`)
     */
    std::string m_inputModule;
};

#endif // MEASUREHANDLER_H
