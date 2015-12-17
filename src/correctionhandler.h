#ifndef CORRECTIONHANDLER_H
#define CORRECTIONHANDLER_H

#include <armadillo>

using namespace arma;

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

class CorrectionHandler
{
public:
    explicit CorrectionHandler();
    int correct(vec &diffX, vec &diffY, bool newInjection, vec &Data_CMx, vec &Data_CMy, int type);
    int checkCorrection();

    void setPID(double P, double I, double D) { m_P = P; m_I = I; m_D = D;};
    void setCMs(vec CMx, vec CMy);
    int numCMx() { m_numCMx; };
    int numCMy() { m_numCMy; };
    void setInjectionCnt(double frequency);
    void setSmat(mat &SmatX, mat &SmatY, double IvecX, double IvecY, bool weightedCorr);

private:
    void initAttributes();
    void initIndexes(double *ADC_WaveIndexX);
    void calcSmat(const arma::mat &Smat, double Ivec, arma::vec &CMWeight, arma::mat &SmatInv);

    int m_injectionCnt;
    int m_injectionStopCnt;
    int m_injectionStartCnt;

    bool m_useCMWeight;
    vec m_CMWeightX, m_CMWeightY;
    int m_numCMx, m_numCMy;
    int m_rmsErrorCnt;

    double m_P, m_I, m_D, m_currentP;
    double m_lastrmsX, m_lastrmsY;

    mat m_SmatInvX, m_SmatInvY;
    vec m_CMx, m_CMy;
    vec m_dCORxPID, m_dCORyPID;
    vec m_dCORlastX, m_dCORlastY;
    vec m_Xsum, m_Ysum;
    vec m_Data_CMx, m_Data_CMy;
};

#endif // CORRECTIONHANDLER_H
