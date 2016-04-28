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
