#ifndef CORRECTIONPROCESSOR_H
#define CORRECTIONPROCESSOR_H

#include "handlers/structures.h"

#include <armadillo>

class ADC;
class RFM;

/**
 * @brief Structure containing every values to process a PID
 */
struct PID_t {
    double P; /**< Gain */
    double I; /**< Coefficient of the integrator */
    double D; /**< Coefficient of the derivator */
    double currentP; /**< Current gain (to modify the P without losing it) */
    arma::vec correctionSum; /**< Sum of the previous corrections */
    arma::vec lastCorrection; /**< Last correction */
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
    int correct(const CorrectionInput_t& input,
                arma::vec& Data_CMx, arma::vec& Data_CMy);

    /**
     * @brief Set the PID parameters.
     */
    void setPID(double P, double I, double D);

    /**
     * @brief Set the correctors.
     */
    void setCMs(arma::vec CMx, arma::vec CMy);

    /**
     * @brief Initialize the injection count.
     *
     *  * count start = frequency/1000
     *  * count stop  = frequency*60/1000
     */
    void setInjectionCnt(double frequency);

    /**
     * @brief Function that call calcSmat() for both x and y axes.
     *
     * @param SmatX, SmatY Matrices to inverse (both axes)
     * @param IvecX, IvecY ????
     * @param CMWeight True if the correction should be weighted or not.
     */
    void setSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr);

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
    arma::vec PIDcorr(const arma::vec& dCM, PID_t& pid);

    Injection_t m_injection; /**< @brief Injection count values */
    int m_rmsErrorCnt; /**< @brief Number of RMS error counted */
    Pair_t<double> m_lastRMS;  /**< @brief Last of RMS */

    bool m_useCMWeight; /**< @brief Should we use weights in the corrections? */
    Pair_t<arma::vec> m_CMWeight; /**< @brief  Weights */
    Pair_t<arma::mat> m_SmatInv; /**< @brief Inverse of the Smatrix */
    Pair_t<PID_t> m_PID; /**< @brief PID parameters */
    Pair_t<arma::vec> m_CM; /** < @brief Current corrector values */
};

#endif // CORRECTIONPROCESSOR_H
