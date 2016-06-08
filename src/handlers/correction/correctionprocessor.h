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
    explicit PID() {};
    PID(const double P, const double I, const double D, const int bufferSize);
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
     * @param[in] diffX BPM values for the x axis
     * @param[in] diffY BPM values for the y axis
     * @param[in] newInjection Was there a new injection?
     * @param[out] Data_CMx Corrector values for the x axis
     * @param[out] Data_CMy Corrector values for the y axis
     * @param[in] type int value in the following set:
     *                  * Correction::None (= `0b00`)
     *                  * Correction::Horizontal (= `0b01`)
     *                  * Correction::Vertical (= `0b10`)
     *                  * Correction::All (= `0b11`)
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
     * @param CMWeight True if the correction should be weighted or not.
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
