#include "correctionprocessor.h"

#include "adc.h"
#include "handlers/handler.h"
#include "logger/logger.h"
#include <iostream>

CorrectionProcessor::CorrectionProcessor()
{
    m_lastrmsX = 999;
    m_lastrmsY = 999;
}

void CorrectionProcessor::setCMs(arma::vec CMx, arma::vec CMy)
{
    m_rmsErrorCnt = 0;

    m_CMx = CMx;
    m_CMy = CMy;

    m_dCORlastX  = arma::zeros<arma::vec>(m_CMx.n_elem);
    m_dCORlastY  = arma::zeros<arma::vec>(m_CMy.n_elem);
    m_Xsum       = arma::zeros<arma::vec>(m_CMx.n_elem);
    m_Ysum       = arma::zeros<arma::vec>(m_CMy.n_elem);
}

void CorrectionProcessor::setSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr)
{
    m_useCMWeight = weightedCorr;
    this->calcSmat(SmatX, IvecX, m_CMWeightX, m_SmatInvX);
    this->calcSmat(SmatY, IvecY, m_CMWeightY, m_SmatInvY);
}

void CorrectionProcessor::setInjectionCnt(double frequency)
{
    m_injectionCnt = 0;
    m_injectionStartCnt = (int) frequency/1000;
    m_injectionStopCnt  = (int) frequency*60/1000;
}

int CorrectionProcessor::correct(const arma::vec &diffX, const arma::vec &diffY,
                                 const bool newInjection,
                                 arma::vec &Data_CMx, arma::vec &Data_CMy,
                                 const int type)
{
    if (sum(diffX) < -10.5) {
        Logger::error(_ME_) << " ERROR: No Beam";
        return FOFB_ERROR_NoBeam;
    }

    if ( newInjection ) {
        m_injectionCnt += 1;
        if ((m_injectionCnt >= m_injectionStopCnt) && (m_injectionCnt <= m_injectionStartCnt)) {
            // We want to write the old value if it is not changed
            Data_CMx = m_CMx;
            Data_CMy = m_CMy;
            return 0;
        }
    }
    m_injectionCnt = 0;


    //cout << "  prove rms" << endl;
    double rmsX = (diffX.n_elem-1) * arma::stddev(diffX) / diffX.n_elem;
    double rmsY = (diffY.n_elem-1) * arma::stddev(diffY) / diffY.n_elem;
    if ((rmsX > m_lastrmsX*1.1) || (rmsY > m_lastrmsY*1.1))
    {
        m_rmsErrorCnt++;
        if (m_rmsErrorCnt > 5) {
            Logger::error(_ME_) << "RMS error, count > 5";
            return FOFB_ERROR_RMS;
        }
    } else {
        m_rmsErrorCnt = 0;
    }
    m_lastrmsX = rmsX;
    m_lastrmsY = rmsY;

    //cout << "  calc dCOR" << endl;
    arma::vec dCMx = m_SmatInvX * diffX;
    arma::vec dCMy = m_SmatInvY * diffY;
    if (m_useCMWeight) {
        dCMx = dCMx % m_CMWeightX;
        dCMy = dCMy % m_CMWeightY;
    }

    if ((max(dCMx) > 0.100) || (max(dCMy) > 0.100))
    {
        Logger::error(_ME_) << "A corrector as a value above 0.100";
        return FOFB_ERROR_CM100;
    }

    //cout << "  calc PID" << endl;
    if (m_currentP < m_P)
        m_currentP += 0.01;

    if ((type & Correction::Horizontal) == Correction::Horizontal) {
        arma::vec dCORxPID = (dCMx * m_currentP) + (m_I*m_Xsum) + (m_D*(dCMx-m_dCORlastX));
        m_dCORlastX = dCMx;
        m_Xsum += m_Xsum + dCMx;
        m_CMx -= dCORxPID;
    }

    if ((type & Correction::Vertical) == Correction::Vertical) {
        arma::vec dCORyPID = (dCMy * m_currentP) + (m_I*m_Ysum) + (m_D*(dCMy-m_dCORlastY));
        m_dCORlastY = dCMy;
        m_Ysum += m_Ysum + dCMy;
        m_CMy -= dCORyPID;
    }

    // We want to write the old value if it is not changed
    Data_CMx = m_CMx;
    Data_CMy = m_CMy;

    return 0;
}

void CorrectionProcessor::calcSmat(const arma::mat &Smat,
                                   double Ivec,
                                   arma::vec &CMWeight,
                                   arma::mat &SmatInv)
{
    Logger::Logger() << "Calculate Smat";
    arma::mat U, S, V;
    arma::vec s;

    Logger::Logger() << "\tGiven : " << " Smat cols: " << Smat.n_cols << " smat rows " << Smat.n_rows << "  Ivec : " << Ivec;
    arma::mat Smat_w = arma::zeros(Smat.n_rows, Smat.n_cols);
    Logger::Logger() << "\tmake Ivec";
    if (Ivec > Smat.n_rows) {
        Logger::Logger() << "\tIVec > Smat.n_rows: Setting Ivec = Smat.n_rows";
        double Ivec = Smat.n_rows;
    }
    if (m_useCMWeight) {
        Logger::Logger() << "\tcalc CMWeights";
        CMWeight = 1/(arma::trans(arma::stddev(Smat)));
        Logger::Logger() << "\tInclude CMWeightX in SMat";
        for (int i = 0; i < Smat.n_cols; i++) {
            Smat_w.col(i) = Smat.col(i) * CMWeight(i);
        }
    } else {
        Smat_w = Smat;
    }
    Logger::Logger() << "\tcalc SVD";
    arma::svd(U,s,V,Smat_w);
    Logger::Logger() << "\treduce U to Ivec";
    U = U.cols(0,Ivec-1);
    Logger::Logger() << "\tTranspose U";
    U = trans(U);
    Logger::Logger() << "\tGet Diag Matrix of  S";
    S = diagmat(s.subvec(0,Ivec-1));
    Logger::Logger() << "\treduce V to Ivec";
    V = V.cols(0,Ivec-1);
    Logger::Logger() << "\tCalc new Matrix";
    SmatInv = V * arma::inv(S) * U;

    Logger::Logger() << "SVD complete ...";
}
