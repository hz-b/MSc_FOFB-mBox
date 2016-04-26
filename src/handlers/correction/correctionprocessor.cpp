#include "correctionprocessor.h"

#include "adc.h"
#include "handlers/handler.h"
#include "logger/logger.h"
#include "logger/messenger.h"

#include <iostream>
#include <cmath>

CorrectionProcessor::CorrectionProcessor()
{
}

void CorrectionProcessor::setCMs(arma::vec CMx, arma::vec CMy)
{
    m_rmsErrorCnt = 0;
    m_lastrmsX = 999;
    m_lastrmsY = 999;

    m_CMx = CMx;
    m_CMy = CMy;

    m_pidX.lastCorrection = arma::zeros<arma::vec>(m_CMx.n_elem);
    m_pidX.correctionSum = arma::zeros<arma::vec>(m_CMx.n_elem);

    m_pidY.lastCorrection = arma::zeros<arma::vec>(m_CMy.n_elem);
    m_pidY.correctionSum = arma::zeros<arma::vec>(m_CMy.n_elem);
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

int CorrectionProcessor::correct(const CorrectionInput_t& input,
                                 arma::vec &Data_CMx, arma::vec &Data_CMy)
{

    if (sum(input.diffX) < -10.5) {
        Logger::error(_ME_) << " ERROR: No Beam";
        return FOFB_ERROR_NoBeam;
    }

    if (this->isInjectionTime(input.newInjection)) {
        // We want to write the old value if it is not changed
        Data_CMx = m_CMx;
        Data_CMy = m_CMy;
        return 0;
    }

    int rmsError = this->checkRMS(input.diffX, input.diffY);
    if (rmsError) {
        return rmsError;
    }

    //cout << "  calc dCOR" << endl;
    arma::vec dCMx = m_SmatInvX * input.diffX;
    arma::vec dCMy = m_SmatInvY * input.diffY;
    if (m_useCMWeight) {
        dCMx = dCMx % m_CMWeightX;
        dCMy = dCMy % m_CMWeightY;
    }

//    if ((arma::max(arma::abs(dCMx)) > 0.100) || (arma::max(arma::abs(dCMy)) > 0.100)) {
    if ((arma::max((dCMx)) > 0.100) || (arma::max((dCMy)) > 0.100)) {
        Logger::error(_ME_) << "A corrector as a value above 0.100";
#ifdef DUMMY_RFM_DRIVER
        Logger::error(_ME_) << "Not considered because DUMMY_RFM_DRIVER is true";
#else
        return FOFB_ERROR_CM100;
#endif
    }

    if ((input.typeCorr & Correction::Horizontal) == Correction::Horizontal) {
        m_CMx -= this->PIDcorr(dCMx, m_pidX);
    }

    if ((input.typeCorr & Correction::Vertical) == Correction::Vertical) {
        m_CMy -= this->PIDcorr(dCMy, m_pidY);
    }
    // We want to write the old value if it is not changed
    Data_CMx = m_CMx;
    Data_CMy = m_CMy;

    arma::vec phaseX10;
    Messenger::get("PHASES-X-10", phaseX10);
    arma::vec ampX10;
    Messenger::get("AMPLITUDES-X-10",ampX10);
    if ((ampX10.n_elem == Data_CMx.n_elem) && (phaseX10.n_elem == Data_CMx.n_elem)) {
        arma::vec dynamicCorrX = ampX10 % (input.value10Hz * arma::cos(phaseX10)
                              + 1-std::pow(input.value10Hz, 2) * arma::sin(phaseX10));
        Data_CMx += dynamicCorrX;
    }

    arma::vec phaseY10;
    Messenger::get("PHASES-Y-10", phaseY10);
    arma::vec ampY10;
    Messenger::get("AMPLITUDES-Y-10",ampY10);
    if ((ampY10.n_elem == Data_CMy.n_elem) && (phaseY10.n_elem == Data_CMy.n_elem)) {
        arma::vec dynamicCorrY = ampY10 % (input.value10Hz * arma::cos(phaseY10)
                              + 1-std::pow(input.value10Hz, 2) * arma::sin(phaseY10));
        Data_CMy += dynamicCorrY;
    }
    return 0;
}

int CorrectionProcessor::checkRMS(const arma::vec& diffX, const arma::vec& diffY)
{
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

    return 0;
}

arma::vec CorrectionProcessor::PIDcorr(const arma::vec& dCM, PID_t& pid)
{
    if (pid.currentP < pid.P) {
        pid.currentP += 0.01;
    }
    arma::vec pidCorrection = (dCM * pid.currentP) + (pid.I*pid.correctionSum) + (pid.D*(dCM-pid.lastCorrection));
    pid.lastCorrection = dCM;
    pid.correctionSum += dCM;

    return pidCorrection;
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

void CorrectionProcessor::setPID(double P, double I, double D)
{
    m_pidX.P = P;
    m_pidX.I = I;
    m_pidX.D = D;
    m_pidX.currentP = 0;

    m_pidY.P = P;
    m_pidY.I = I;
    m_pidY.D = D;
    m_pidX.currentP = 0;
}

bool CorrectionProcessor::isInjectionTime(const bool newInjection)
{
    if ( newInjection ) {
        m_injectionCnt += 1;
        if ((m_injectionCnt >= m_injectionStopCnt) && (m_injectionCnt <= m_injectionStartCnt)) {
            return true;
        }
    } else {
        m_injectionCnt = 0;
    }
    return false;
}
