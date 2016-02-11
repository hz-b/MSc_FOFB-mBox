#ifndef CORRECTIONPROCESSOR_H
#define CORRECTIONPROCESSOR_H

#include <armadillo>

namespace Correction {
    enum Type {
        None = 0b01,
        Horizontal = 0b01,
        Vertical = 0b10,
        All = 0b11
    };
}

class ADC;
class RFM;

class CorrectionProcessor
{
public:
    explicit CorrectionProcessor();
    int correct(arma::vec &diffX, arma::vec &diffY, bool newInjection, arma::vec &Data_CMx, arma::vec &Data_CMy, int type);
    int checkCorrection();

    void setPID(double P, double I, double D) { m_P = P; m_I = I; m_D = D;};
    void setCMs(arma::vec CMx, arma::vec CMy);
    int numCMx() { return m_CMx.n_elem; };
    int numCMy() { return m_CMy.n_elem; };
    void setInjectionCnt(double frequency);
    void setSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr);

private:
    void calcSmat(const arma::mat &Smat, double Ivec, arma::vec &CMWeight, arma::mat &SmatInv);

    int m_injectionCnt;
    int m_injectionStopCnt;
    int m_injectionStartCnt;

    bool m_useCMWeight;
    arma::vec m_CMWeightX, m_CMWeightY;
    int m_rmsErrorCnt;

    double m_P, m_I, m_D, m_currentP;
    double m_lastrmsX, m_lastrmsY;

    arma::mat m_SmatInvX, m_SmatInvY;
    arma::vec m_CMx, m_CMy;
    arma::vec m_dCORxPID, m_dCORyPID;
    arma::vec m_dCORlastX, m_dCORlastY;
    arma::vec m_Xsum, m_Ysum;
    arma::vec m_Data_CMx, m_Data_CMy;
};

#endif // CORRECTIONPROCESSOR_H
