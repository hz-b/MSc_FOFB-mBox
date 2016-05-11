#include "handlers/measures/measurehandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "handlers/structures.h"
#include "modules/zmq/logger.h"

#include <iostream>
#include <string>
#include <math.h>
#include <Python.h>

/**
 * @brief Just to remove some compiler warnings.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


/**
 * @brief: Hack: using import_array() with Python2 doesn't work... We should
 * definetely move to Python3
 */
#define my_import_array() {int r =_import_array();  \
if (r < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");} }

MeasureHandler::MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr,
                                std::string inputFile)
    : Handler(driver, dma, weightedCorr)
{
    m_inputFile = inputFile;
    int timeValue = 0;

    this->setModule();
}


MeasureHandler::~MeasureHandler()
{
    Py_XDECREF(m_pModule);
    Py_XDECREF(m_pFunc);
    Py_Finalize();
}

int MeasureHandler::typeCorrection()
{
    return Correction::All;
}

int MeasureHandler::callProcessorRoutine(const CorrectionInput_t& input,
                                         arma::vec& CMx, arma::vec& CMy)
{

    int error = this->callPythonFunction(input.diff.x, input.diff.x, CMx, CMy);

    // Add to init values
    CMx += m_CM.x;
    CMy += m_CM.y;

    return error;
}

void MeasureHandler::setProcessor(arma::mat SmatX, arma::mat SmatY,
                                  double IvecX, double IvecY,
                                  double Frequency,
                                  double P, double I, double D,
                                  arma::vec CMx, arma::vec CMy,
                                  bool weightedCorr)
{
    int errorPythonInit = this->initPython();
    m_CM.x = CMx;
    m_CM.y = CMy;

    std::cout << m_CM.x[1] << '\n';
    if (errorPythonInit) {
        Logger::error(_ME_) << "error";
    }
}

/**
 * Let's say that m_inputFile = `/PATH/TO/FILE.py`
 * 1. Separate `PATH/TO` and `FILE`
 * 2. Set `PATH/TO` in the path of python
 * 3. Define `FILE.py` as the module name
 * 4. Remove `.py` from the module name
 */
void MeasureHandler::setModule()
{
    std::size_t found = m_inputFile.find_last_of("/");
    m_inputPath = m_inputFile.substr(0, found);
    m_inputModule = m_inputFile.substr(found+1);

    std::string suffix = ".py";
    found = m_inputModule.find_last_of(".");
    m_inputModule = m_inputModule.substr(0, found);
}

/**
 * @see https://docs.python.org/2/extending/embedding.html
 */
int MeasureHandler::initPython()
{
    PyObject *pName = NULL;

    Py_InitializeEx(0);

#if PY_MAJOR_VERSION  ==  2
    my_import_array();
#elif PY_MAJOR_VERSION  ==  3
    import_array();
#endif

    PyRun_SimpleString("import sys");
    std::string cmd = "sys.path.append(\"" + m_inputPath + "\")";
    PyRun_SimpleString(cmd.c_str());

    pName = PyUnicode_FromString(m_inputModule.c_str());

    m_pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (m_pModule != NULL) {
        m_pFunc = PyObject_GetAttrString(m_pModule, PYTHON_CORRECTION_FUNCTION.c_str());

        if (m_pFunc && PyCallable_Check(m_pFunc)) {
            return callPythonInit();
        } else {
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            Logger::error(_ME_) << "Cannot find function '"<< PYTHON_CORRECTION_FUNCTION <<"'";
        }
        return 1;
    } else {
        PyErr_Print();
        Logger::error(_ME_) << "Failed to load " << m_inputFile;
        return 1;
    }
}

int MeasureHandler::callPythonInit()
{
    PyObject *pArgs = NULL;
    PyObject *pFunc = PyObject_GetAttrString(m_pModule, "init");

    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject *pyBPMx_nb = PyLong_FromLong(m_numBPM.x);
        PyObject *pyBPMy_nb = PyLong_FromLong(m_numBPM.y);
        PyObject *pyCMx_nb = PyLong_FromLong(m_numCM.x);
        PyObject *pyCMy_nb = PyLong_FromLong(m_numCM.y);

        Py_INCREF(pyBPMx_nb);
        Py_INCREF(pyBPMy_nb);
        Py_INCREF(pyCMx_nb);
        Py_INCREF(pyCMy_nb);

        if (!pyBPMx_nb || !pyBPMy_nb || !pyCMx_nb || !pyCMy_nb) {
            Py_DECREF(pyBPMx_nb);
            Py_DECREF(pyBPMy_nb);
            Py_DECREF(pyCMy_nb);
            Py_DECREF(pyCMy_nb);
            Logger::error(_ME_) << "Cannot convert arguments";
            return 1;
        }
        pArgs = PyTuple_New(4);
        // pValue reference stolen here:
        PyTuple_SetItem(pArgs, 0, pyBPMx_nb);
        PyTuple_SetItem(pArgs, 1, pyBPMy_nb);
        PyTuple_SetItem(pArgs, 2, pyCMx_nb);
        PyTuple_SetItem(pArgs, 3, pyCMy_nb);

        PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);

        // Everything must be unreferenced
        Py_DECREF(pyCMx_nb);
        Py_DECREF(pyCMy_nb);
        Py_DECREF(pyBPMx_nb);
        Py_DECREF(pyBPMy_nb);

        return 0;

    } else {
        PyErr_Print();
        Logger::error(_ME_) << "Call failed";
        return 1;
    }
}


int MeasureHandler::callPythonFunction(const arma::vec& BPMx, const arma::vec& BPMy,
                                       arma::vec& CMx, arma::vec& CMy)
{
    PyObject *pArgs = NULL, *pValue = NULL;

    npy_intp BPMx_s[] = {m_numBPM.x};
    npy_intp BPMy_s[] = {m_numBPM.y};
    npy_intp CMx_s[] = {m_numCM.x};
    npy_intp CMy_s[] = {m_numCM.y};

    PyObject *pyBPMx = PyArray_SimpleNewFromData(1, BPMx_s,
                                                 NPY_DOUBLE, (double*) BPMx.memptr());
    PyObject *pyBPMy = PyArray_SimpleNewFromData(1, BPMy_s,
                                                 NPY_DOUBLE, (double*) BPMy.memptr());

    Py_INCREF(pyBPMx);
    Py_INCREF(pyBPMy);

    if (!pyBPMx || !pyBPMy) {
        Py_DECREF(pyBPMx);
        Py_DECREF(pyBPMy);
        Logger::error(_ME_) << "Cannot convert argument";
        return 1;
    }

    pArgs = PyTuple_New(2);
    // pValue reference stolen here:
    PyTuple_SetItem(pArgs, 0, pyBPMx);
    PyTuple_SetItem(pArgs, 1, pyBPMy);

    pValue = PyObject_CallObject(m_pFunc, pArgs);

    Py_DECREF(pArgs);

    if (pValue != NULL && PyTuple_Size(pValue) == 2) {
        PyArrayObject *pyCMx = (PyArrayObject*)PyTuple_GetItem(pValue, 0);
        PyArrayObject *pyCMy = (PyArrayObject*)PyTuple_GetItem(pValue, 1);

        // use the constructor vec(ptr, nb_elements)
        // with ptr = (double*) PyArray_DATA(pyCMx)
        CMx = arma::vec((double*) PyArray_DATA(pyCMx), m_numCM.x);
        CMy = arma::vec((double*) PyArray_DATA(pyCMy), m_numCM.y);

        // Everything must be unreferenced
        Py_DECREF(pValue);
        Py_DECREF(pyBPMx);
        Py_DECREF(pyBPMy);

        return 0;

    } else {
        PyErr_Print();
        Logger::error(_ME_) << "Call failed";
        return 1;
    }
}
