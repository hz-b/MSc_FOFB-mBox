#include "dynamic10hzcorrectionprocessor.h"

#include <algorithm>

#include "modules/zmq/logger.h"
#include "modules/zmq/messenger.h"

Dynamic10HzCorrectionProcessor::Dynamic10HzCorrectionProcessor()
{

}

void Dynamic10HzCorrectionProcessor::initialize()
{
    m_started = false;
    m_buffer10Hz.zeros();
}

int Dynamic10HzCorrectionProcessor::process(const CorrectionInput_t& input,
                                            arma::vec& Data_CMx, arma::vec& Data_CMy)
{
    this->updateBuffer10Hz(input.value10Hz);

    int errorX = this->processAxis("x", Data_CMx);
    int errorY = this->processAxis("y", Data_CMy);

    return (errorX | errorY);
}

void Dynamic10HzCorrectionProcessor::updateBuffer10Hz(const double newValue){
     // Pop front element, them pushback new one
    for (int i = 0 ; i < NTAPS - 1 ; i++) {
        m_buffer10Hz(i) = m_buffer10Hz(i+1);
    }
    m_buffer10Hz(NTAPS-1) = newValue;
}


int Dynamic10HzCorrectionProcessor::processAxis(const std::string& axisName,
                                                      arma::vec& outputData)
{
    std::string axis = axisName;
    std::transform(axis.begin(), axis.end(), axis.begin(), toupper);
    int vectorSize = outputData.n_elem;

    double ampref;
    double phref;
    Messenger::get("AMPLITUDE-REF-10", ampref);
    Messenger::get("PHASE-REF-10", phref);

    arma::vec phase;
    Messenger::get("PHASES-"+axis+"-10", phase);
    arma::vec amp;
    Messenger::get("AMPLITUDES-"+axis+"-10", amp);

    // Dynamic correction values are set.
    if (amp.empty() || phase.empty()) {
        return 0;  // It's not an error.
    }

    // Size is ok
    if ((amp.n_elem != vectorSize) || (phase.n_elem != vectorSize)) {
        Logger::error(_ME_) << "Dynamic correction: size not correct.";
        return 1;
    }

    if (!m_started) {
        m_started = true;
        Logger::Logger() << "Dynamic correction started.";
    }

    arma::vec time = arma::linspace<arma::vec>(0, NTAPS-1, NTAPS);
    arma::mat t_mat = arma::repmat(time.t(), vectorSize, 1)/SAMPLING_FREQ;

    arma::mat phase_mat = arma::repmat(phase, 1, NTAPS) - phref;
    arma::mat fir = arma::cos(2*M_PI*FREQ*t_mat - phase_mat)*2/NTAPS; // - or + the phase ???


    arma::vec dynamicCorr = arma::zeros<arma::vec>(vectorSize);

    for (int i = 0 ; i < vectorSize ; i++){
        dynamicCorr.row(i) = fir.row(i) * arma::flipud(m_buffer10Hz);
    }

    dynamicCorr %= amp/ampref;

    // Check amplitude before applying
    if ((arma::max(arma::abs(dynamicCorr)) > 0.1)) {
        Logger::error(_ME_) << "Dynamic amplitude to high, don't use";
        return 1;
    }
    std::cout << arma::max(arma::abs(dynamicCorr)) << '\n'<<'\n';
    outputData += dynamicCorr;

    return 0;
}
