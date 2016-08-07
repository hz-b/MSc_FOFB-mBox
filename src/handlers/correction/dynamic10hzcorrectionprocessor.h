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

#ifndef DYNAMIC10HZCORRECTIONPROCESSOR_H
#define DYNAMIC10HZCORRECTIONPROCESSOR_H

#include <armadillo>

#include "handlers/structures.h"

const int NTAPS = 15; /**< @brief Tap number for the FIR filter */

/**
 * @brief Do a correction for the 10Hz harmonic perturbation.
 *
 * The correction uses a FIR filter
 *      \f[
 *          h = \cos(2 \pi 10 t + \phi)
 *      \f]
 * where \f$\phi\f$ is the phase change to apply, \f$t\f$ is the sampled duration
 * of 0.1s, at frequency 150Hz. This produces exactly one period, and the FIR as
 * 15 taps.
 */
class Dynamic10HzCorrectionProcessor
{

const double SAMPLING_FREQ = 150; /**< @brief Sampling frequency */
const double FREQ = 10; /**< @brief Frequency of interest */

public:
    /**
     * @brief Constructor. Does nothing particular
     */
    Dynamic10HzCorrectionProcessor();

    /**
     * @brief Initialize attributes.
     */
    void initialize();

    /**
     * @brief Do the full correction.
     *
     * This calls processAxis() for each axis ('x' and 'y').
     *
     * @param input Input values to correct
     * @param Data_CMx Correction output (horizontal axis)
     * @param Data_CMy Correction output (vertical axis)
     *
     * @return 1 if an error occurs, 0 else
     */
    int process(const CorrectionInput_t& input,
                arma::vec& Data_CMx, arma::vec& Data_CMy);

private:
    /**
     * @brief Stack last 10 Hz value (pushback), dequeue oldest one (popfront).
     */
    void updateBuffer10Hz(const double newValue);

    /**
     * @brief Actually do the correction on a given axis.
     *
     * @param axis Name of the axis: 'x' or 'y'
     * @param outputData Corresponding data to process. The result is added to
     *                   the input value.
     * @return 1 if an error occured, 0 else.
     */
    int processAxis(const std::string& axis, arma::vec& outputData);

    arma::vec::fixed<NTAPS> m_buffer10Hz; /**< @brief Buffer containing the last 10Hz values */
    bool m_started;  /**< @brief Flag to know whether the correction has started or not */
};

#endif // DYNAMIC10HZCORRECTIONPROCESSOR_H
