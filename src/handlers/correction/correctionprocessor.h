#ifndef CORRECTIONPROCESSOR_H
#define CORRECTIONPROCESSOR_H

#include "handlers/structures.h"

#include <armadillo>

class ADC;
class RFM;


/**
 * @brief Class containing every values to process a PID
 */
class PID {
public:
    /**
     * @brief Empty constructor
     */
    explicit PID() { PID(0., 0., 0., 0); }

    /**
     * @brief Real constructor
     * @param P Gain
     * @param I Integrator coefficient
     * @param D Derivator coefficient
     * @param bufferSize Size of the buffers (= size of input/output vectors)
     */
    PID(const double P, const double I, const double D, const int bufferSize);

    /**
     * @brief Apply the PID to a data vector
     * @param dCM Data on which the PID should be applied
     * @return Computed vector
     */
    arma::vec apply(const arma::vec& dCM);

private:
    double m_P; /**< Gain */
    double m_I; /**< Coefficient of the integrator */
    double m_D; /**< Coefficient of the derivator */
    double m_currentP; /**< Current gain (to modify the P without losing it) */
    arma::vec m_correctionSum; /**< Sum of the previous corrections */
    arma::vec m_lastCorrection; /**< Last correction */
};

/**
 * @brief Structure containing values concerning the injection
 */
struct Injection_t {
    int count; /**< Current number of injection since reset */
    int countStart; /**< Number of injection, low threshold */
    int countStop; /**< Number of injection, high threshold */
};


/**
 * @brief Do the correction.
 *
 * The correction uses the pseudo-inverse of the ring response matrix, weighten
 * the output and applieds a PID.
 */
class CorrectionProcessor
{
public:
    /**
     * @brief Constructor
     */
    explicit CorrectionProcessor();

    /**
     * @brief Calculate the correction to apply.
     *
     * @param[in] input All values needed as input for the correction
     * @param[out] Data_CMx Corrector values for the x axis
     * @param[out] Data_CMy Corrector values for the y axis
     */
    int process(const CorrectionInput_t& input,
                arma::vec& Data_CMx, arma::vec& Data_CMy);

    /**
     * @brief Set the PID parameters.
     */
    void initPID(double P, double I, double D);

    /**
     * @brief Set the correctors.
     */
    void initCMs(arma::vec CMx, arma::vec CMy);

    /**
     * @brief Initialize the injection count.
     *
     *  * count start = frequency/1000
     *  * count stop  = frequency*60/1000
     */
    void initInjectionCnt(double frequency);

    /**
     * @brief To be called after all other parameters are initialized.
     */
    void finishInitialization();

    /**
     * @brief Function that call calcSmat() for both x and y axes.
     *
     * @param SmatX, SmatY Matrices to inverse (both axes)
     * @param IvecX, IvecY ????
     * @param weightedCorr True if the correction should be weighted or not.
     */
    void initSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr);

private:

    /**
     * @brief Calculate the inverse of the S matrix.
     *
     * @param[in] Smat Matrix to inverse
     * @param[in] Ivec ????
     * @param[out] CMWeight Vector of weight to apply for each corrector
     * @param[out] SmatInv Inversed matrix
     */
    void calcSmat(const arma::mat &Smat, double Ivec, arma::vec &CMWeight, arma::mat &SmatInv);

    bool isInjectionTime(const bool newInjection);
    int checkRMS(const arma::vec& diffX, const arma::vec& diffY);

    Injection_t m_injection; /**< @brief Injection count values */
    int m_rmsErrorCnt; /**< @brief Number of RMS error counted */
    Pair_t<double> m_lastRMS;  /**< @brief Last of RMS */

    bool m_useCMWeight; /**< @brief Should we use weights in the corrections? */
    Pair_t<arma::vec> m_CMWeight; /**< @brief  Weights */
    Pair_t<arma::mat> m_SmatInv; /**< @brief Inverse of the Smatrix */
    Pair_t<PID> m_PID; /**< @brief PID classes */
    Pair_t<arma::vec> m_CM; /** < @brief Current corrector values */
};

#endif // CORRECTIONPROCESSOR_H
