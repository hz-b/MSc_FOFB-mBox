/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
