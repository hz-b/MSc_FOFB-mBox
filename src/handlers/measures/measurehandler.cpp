#include <Python.h> // Should be first to remove warnings

#include "handlers/measures/measurehandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"

#include <iostream>
#include <string>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// HACK: using import_array() with Python2 doesn't work... We should definetely move to Python3
#define my_import_array() {int r =_import_array(); std::cout <<r << std::endl; if (r < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");} }

MeasureHandler::MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr,
                                std::string inputFile)
    : Handler(driver, dma, weightedCorr)
{
    m_inputFile = inputFile;
    m_functionName = "corr_value";
    int timeValue = 0;

    this->setModule();
    int errorPythonInit = this->initPython();
    if (errorPythonInit) {
        std::cout << "error in python Init" << std::endl;
        m_status = Error;
    }
}


MeasureHandler::~MeasureHandler()
{
    Py_XDECREF(m_pFunc);
    Py_Finalize();
}


int MeasureHandler::make()
{
    arma::vec diffX, diffY;
    arma::vec CMx, CMy;
    bool newInjection;
    this->getNewData(diffX, diffY, newInjection);
    
    /// SAVE diffX diffY and get amp
    
    int error = this->callPythonFunction(diffX, diffY, CMx, CMy);

    return 0;
}


void MeasureHandler::setProcessor(arma::mat SmatX, arma::mat SmatY,
                                     double IvecX, double IvecY,
                                     double Frequency, 
                                     double P, double I, double D,
                                     arma::vec CMx, arma::vec CMy,
                                     bool weightedCorr)
{
}

/**
 * Let's say that m_inputFile = /PATH/TO/FILE.py
 * 1) Separate PATH/TO and FILE
 * 2) Set PATH/TO in the path of python
 * 3) Define FILE.py as the module name
 * 4) Remove ".py" from the module name
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
 * @errors 0 if success, 1 if failure
 * @see https://docs.python.org/2/extending/embedding.html
 */
int MeasureHandler::initPython()
{
    PyObject *pName, *pModule, *pDict;

    Py_SetProgramName("");
    Py_InitializeEx(0);
    my_import_array();
    PyRun_SimpleString("import sys");
    std::string cmd = "sys.path.append(\"" + m_inputPath + "\")";
    PyRun_SimpleString(cmd.c_str());

    pName = PyUnicode_FromString(m_inputModule.c_str());

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        m_pFunc = PyObject_GetAttrString(pModule, m_functionName.c_str());
        // pFunc is a new reference
        Py_DECREF(pModule);

        if (m_pFunc && PyCallable_Check(m_pFunc)) {
            return 0;
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            std::cout << "Cannot find function '"<< m_functionName <<"'" << std::endl;
        }
        Py_DECREF(pModule);
        return 1;
    } else {
        PyErr_Print();
        std::cout << "Failed to load " << m_inputFile << std::endl;
        return 1;
    }
}

int MeasureHandler::callPythonFunction(const arma::vec &BPMx, const arma::vec &BPMy,
                                       arma::vec &CMx, arma::vec &CMy)
{
    PyObject *pArgs = NULL, *pValue = NULL;

    pArgs = PyTuple_New(4);

    npy_intp BPMx_s[] = {m_numBPMx};
    npy_intp BPMy_s[] = {m_numBPMy};
    npy_intp CMx_s[] = {m_numCMx};
    npy_intp CMy_s[] = {m_numCMy};
    
    PyObject *pyBPMx = PyArray_SimpleNewFromData(1, BPMx_s,
                                                 NPY_DOUBLE, (void*) BPMx.memptr());
    PyObject *pyBPMy = PyArray_SimpleNewFromData(1, BPMy_s,
                                                 NPY_DOUBLE, (void*) BPMy.memptr());
    PyObject *pyCMx_nb = PyLong_FromLong(m_numCMx);
    PyObject *pyCMy_nb = PyLong_FromLong(m_numCMy);
    
    Py_INCREF(pyBPMx);
    Py_INCREF(pyBPMy);
    Py_INCREF(pyCMx_nb);
    Py_INCREF(pyCMy_nb);
    
    if (!pyBPMx || !pyBPMy) {
        Py_DECREF(pyBPMx);
        Py_DECREF(pyBPMy);
        std::cout << "Cannot convert argument" << std::endl;
        return 1;
    }
    // pValue reference stolen here:
    PyTuple_SetItem(pArgs, 0, pyBPMx);
    PyTuple_SetItem(pArgs, 1, pyBPMy);
    PyTuple_SetItem(pArgs, 2, pyCMx_nb);
    PyTuple_SetItem(pArgs, 3, pyCMy_nb);

    pValue = PyObject_CallObject(m_pFunc, pArgs);
    Py_DECREF(pArgs);
    if (pValue != NULL && PyTuple_Size(pValue) == 2) {
        PyArrayObject *pyCMx = (PyArrayObject*)PyTuple_GetItem(pValue, 0);
        PyArrayObject *pyCMy = (PyArrayObject*)PyTuple_GetItem(pValue, 1);
        
        /**
         * use the constructor vec(ptr, nb_elements)
         * ptr = (double*) PyArray_GetPtr(pyCMx, &CMx_s)
         */
        CMx = arma::vec((double*) PyArray_DATA(pyCMx), m_numCMx);
        CMy = arma::vec((double*) PyArray_DATA(pyCMx), m_numCMy);
            std::cout << CMx[20] << std::endl;
        /**
         * Everything must be unreferenced
         */
        //Py_DECREF(pValue);
        Py_DECREF(pyCMx);
        Py_DECREF(pyCMy);
        Py_DECREF(pyBPMx);
        Py_DECREF(pyBPMy);
        
        return 0;
        
    } else {
        PyErr_Print();
        std::cout << "Call failed" << std::endl;
        return 1;
    }
}