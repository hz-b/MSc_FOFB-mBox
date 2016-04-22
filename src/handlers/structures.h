#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <armadillo>

struct CorrectionInput_t {
    arma::vec diffX;
    arma::vec diffY;
    bool newInjection;
    int typeCorr;
    double value10Hz;
};

#endif //STRUCTURES_H
