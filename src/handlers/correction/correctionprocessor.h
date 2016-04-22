#ifndef CORRECTIONPROCESSOR_H
#define CORRECTIONPROCESSOR_H

#include "handlers/structures.h"

#include <armadillo>

class ADC;
class RFM;

struct PID_t {
    double P;
    double I;
    double D;
    double currentP;
    arma::vec correctionSum;
    arma::vec lastCorrection;
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


    int m_injectionCnt;
    int m_injectionStopCnt;
    int m_injectionStartCnt;
    int m_rmsErrorCnt;

    bool m_useCMWeight;
    arma::vec m_CMWeightX, m_CMWeightY;

    double m_P, m_I, m_D, m_currentP;
    double m_lastrmsX, m_lastrmsY;

    arma::mat m_SmatInvX, m_SmatInvY;
    arma::vec m_CMx, m_CMy;
    PID_t m_pidX;
    PID_t m_pidY;
    arma::vec m_dCORlastX, m_dCORlastY;
    arma::vec m_Xsum, m_Ysum;
    arma::vec m_Data_CMx, m_Data_CMy;
};

#endif // CORRECTIONPROCESSOR_H
