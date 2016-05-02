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
    m_lastRMS.x = 999;
    m_lastRMS.y = 999;

    m_CM.x = CMx;
    m_CM.y = CMy;

    m_PID.x.lastCorrection = arma::zeros<arma::vec>(m_CM.x.n_elem);
    m_PID.x.correctionSum = arma::zeros<arma::vec>(m_CM.x.n_elem);

    m_PID.y.lastCorrection = arma::zeros<arma::vec>(m_CM.y.n_elem);
    m_PID.y.correctionSum = arma::zeros<arma::vec>(m_CM.y.n_elem);
}

void CorrectionProcessor::setSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr)
{
    m_useCMWeight = weightedCorr;
    this->calcSmat(SmatX, IvecX, m_CMWeight.x, m_SmatInv.x);
    this->calcSmat(SmatY, IvecY, m_CMWeight.y, m_SmatInv.y);
}

void CorrectionProcessor::setInjectionCnt(double frequency)
{
    m_injection.count = 0;
    m_injection.countStart = (int) frequency/1000;
    m_injection.countStop  = (int) frequency*60/1000;
}

int CorrectionProcessor::correct(const CorrectionInput_t& input,
                                 arma::vec &Data_CMx, arma::vec &Data_CMy)
{

    if (sum(input.diff.x) < -10.5) {
        Logger::error(_ME_) << " ERROR: No Beam";
        return FOFB_ERROR_NoBeam;
    }

    if (this->isInjectionTime(input.newInjection)) {
        // We want to write the old value if it is not changed
        Data_CMx = m_CM.x;
        Data_CMy = m_CM.y;
        return 0;
    }

    int rmsError = this->checkRMS(input.diff.x, input.diff.y);
    if (rmsError) {
        return rmsError;
    }

    //cout << "  calc dCOR" << endl;
    arma::vec dCMx = m_SmatInv.x * input.diff.x;
    arma::vec dCMy = m_SmatInv.y * input.diff.y;
    if (m_useCMWeight) {
        dCMx = dCMx % m_CMWeight.x;
        dCMy = dCMy % m_CMWeight.y;
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
        m_CM.x -= this->PIDcorr(dCMx, m_PID.x);
    }

    if ((input.typeCorr & Correction::Vertical) == Correction::Vertical) {
        m_CM.y -= this->PIDcorr(dCMy, m_PID.y);
    }
    // We want to write the old value if it is not changed
    Data_CMx = m_CM.x;
    Data_CMy = m_CM.y;

    double ampref;
    double phref;
    Messenger::get("AMPLITUDE-REF-10", ampref);
    Messenger::get("PHASE-REF-10", phref);

    arma::vec phaseX10;
    Messenger::get("PHASES-X-10", phaseX10);
    arma::vec ampX10;
    Messenger::get("AMPLITUDES-X-10", ampX10);
    if ((ampX10.n_elem == Data_CMx.n_elem) && (phaseX10.n_elem == Data_CMx.n_elem)) {
        arma::vec dynamicCorrX = ampX10 % (input.value10Hz/ampref * arma::cos(phaseX10-phref)
                              + std::sqrt(1-std::pow(input.value10Hz/ampref, 2)) * arma::sin(phaseX10-phref));
        Data_CMx += dynamicCorrX;
    }

    arma::vec phaseY10;
    Messenger::get("PHASES-Y-10", phaseY10);
    arma::vec ampY10;
    Messenger::get("AMPLITUDES-Y-10",ampY10);
    if ((ampY10.n_elem == Data_CMy.n_elem) && (phaseY10.n_elem == Data_CMy.n_elem)) {
        arma::vec dynamicCorrY = ampY10 % (input.value10Hz/ampref * arma::cos(phaseY10-phref)
                              + std::sqrt(1-std::pow(input.value10Hz/ampref, 2)) * arma::sin(phaseY10-phref));
        Data_CMy += dynamicCorrY;
    }
    return 0;
}

int CorrectionProcessor::checkRMS(const arma::vec& diffX, const arma::vec& diffY)
{
    double rmsX = (diffX.n_elem-1) * arma::stddev(diffX) / diffX.n_elem;
    double rmsY = (diffY.n_elem-1) * arma::stddev(diffY) / diffY.n_elem;
    if ((rmsX > m_lastRMS.x*1.1) || (rmsY > m_lastRMS.y*1.1))
    {
        m_rmsErrorCnt++;
        if (m_rmsErrorCnt > 5) {
            Logger::error(_ME_) << "RMS error, count > 5";
            return FOFB_ERROR_RMS;
        }
    } else {
        m_rmsErrorCnt = 0;
    }

    m_lastRMS.x = rmsX;
    m_lastRMS.y = rmsY;

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
    m_PID.x.P = P;
    m_PID.x.I = I;
    m_PID.x.D = D;
    m_PID.x.currentP = 0;

    m_PID.y.P = P;
    m_PID.y.I = I;
    m_PID.y.D = D;
    m_PID.y.currentP = 0;
}

bool CorrectionProcessor::isInjectionTime(const bool newInjection)
{
    if ( newInjection ) {
        m_injection.count += 1;
        if ((m_injection.count >= m_injection.countStop) && (m_injection.count <= m_injection.countStart)) {
            return true;
        }
    } else {
        m_injection.count = 0;
    }
    return false;
}
