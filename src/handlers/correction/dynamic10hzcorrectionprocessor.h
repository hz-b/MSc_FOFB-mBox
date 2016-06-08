#ifndef DYNAMIC10HZCORRECTIONPROCESSOR_H
#define DYNAMIC10HZCORRECTIONPROCESSOR_H

#include <armadillo>

#include "handlers/structures.h"

const int NTAPS = 15;

class Dynamic10HzCorrectionProcessor
{
const double SAMPLING_FREQ = 150;
const double FREQ = 10;

public:
    Dynamic10HzCorrectionProcessor();
    void initialize();
    int process(const CorrectionInput_t& input,
                arma::vec& Data_CMx, arma::vec& Data_CMy);

private:
    void updateBuffer10Hz(const double newValue);
    int processAxis(const std::string& axis, arma::vec& outputData);

    arma::vec::fixed<NTAPS> m_buffer10Hz;
    bool m_started;
};

#endif // DYNAMIC10HZCORRECTIONPROCESSOR_H
