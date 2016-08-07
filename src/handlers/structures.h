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

#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <armadillo>

/**
 * @brief Structure containing x and y values of a same type.
 */
template <typename T> struct Pair_t {
    T x; /**< X-value */
    T y; /**< Y-value */
};

/**
 * @brief Input values needed for a correction
 */
struct CorrectionInput_t {
    Pair_t<arma::vec> diff; /**< @brief Differential orbit */
    bool newInjection; /**< @brief Is there a new injection? */
    int typeCorr; /**< @brief Type of correction to apply */
    double value10Hz; /**< @brief Last current value of the 10Hz magnet */
};

#endif //STRUCTURES_H
