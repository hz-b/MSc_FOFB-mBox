#include "correctionprocessor.h"

#include "adc.h"
#include "handlers/handler.h"
#include "modules/zmq/logger.h"
#include "modules/zmq/messenger.h"

#include <iostream>
#include <cmath>

const int FIR_NTAP = 15;

CorrectionProcessor::CorrectionProcessor()
{
}

void CorrectionProcessor::initCMs(arma::vec CMx, arma::vec CMy)
{
    m_CM.x = CMx;
    m_CM.y = CMy;
}
void CorrectionProcessor::finishInitialization()
{
    m_rmsErrorCnt = 0;
    m_lastRMS.x = 999;
    m_lastRMS.y = 999;

    m_PID.x.lastCorrection = arma::zeros<arma::vec>(m_CM.x.n_elem);
    m_PID.x.correctionSum = arma::zeros<arma::vec>(m_CM.x.n_elem);

    m_PID.y.lastCorrection = arma::zeros<arma::vec>(m_CM.y.n_elem);
    m_PID.y.correctionSum = arma::zeros<arma::vec>(m_CM.y.n_elem);

    m_values10Hz = arma::zeros<arma::vec>(FIR_NTAP);
    m_dynamicCorrStarted = false;
}

void CorrectionProcessor::initSmat(arma::mat &SmatX, arma::mat &SmatY, double IvecX, double IvecY, bool weightedCorr)
{
    m_useCMWeight = weightedCorr;
    this->calcSmat(SmatX, IvecX, m_CMWeight.x, m_SmatInv.x);
    this->calcSmat(SmatY, IvecY, m_CMWeight.y, m_SmatInv.y);
}

void CorrectionProcessor::initPID(double P, double I, double D)
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

void CorrectionProcessor::initInjectionCnt(double frequency)
{
    m_injection.count = 0;
    m_injection.countStart = (int) frequency/1000;
    m_injection.countStop  = (int) frequency*60/1000;
}

int CorrectionProcessor::correct(const CorrectionInput_t& input,
                                 arma::vec &Data_CMx, arma::vec &Data_CMy)
{

    if (sum(input.diff.x) < -10.5) {
#ifndef DUMMY_RFM_DRIVER
        Logger::error(_ME_) << " ERROR: No Beam";
        return FOFB_ERROR_NoBeam;
#else
        //Logger::error(_ME_) << "Not considered because DUMMY_RFM_DRIVER is true";
#endif
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

    if ((arma::max(arma::abs(dCMx)) > 0.100) || (arma::max(arma::abs(dCMy)) > 0.100)) {
//    if ((arma::max((dCMx)) > 0.100) || (arma::max((dCMy)) > 0.100)) {

#ifndef DUMMY_RFM_DRIVER
        Logger::error(_ME_) << "A corrector as a value above 0.100";
        return FOFB_ERROR_CM100;
#else
        //Logger::error(_ME_) << "Not considered because DUMMY_RFM_DRIVER is true";
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

    Data_CMx += this->dynamicCorrection10("X", Data_CMx.n_elem, input.value10Hz);
    Data_CMy += this->dynamicCorrection10("Y", Data_CMy.n_elem, input.value10Hz);
    return 0;
}

arma::vec CorrectionProcessor::dynamicCorrection10(const std::string& axis,
                                                   const int size, const double value10Hz)
{
    double ampref;
    double phref;
    Messenger::get("AMPLITUDE-REF-10", ampref);
    Messenger::get("PHASE-REF-10", phref);

    arma::vec phase;
    Messenger::get("PHASES-"+axis+"-10", phase);
    arma::vec amp;
    Messenger::get("AMPLITUDES-"+axis+"-10", amp);

    if (amp.empty() || phase.empty()) {
        return arma::zeros<arma::vec>(size);
    }

    if ((arma::max(arma::abs(amp)) > 0.1) || (arma::max(arma::abs(amp)) > 0.1) || ampref < 1e-6) {
        Logger::error(_ME_) << "Dynamic amplitude to high, don't use";
        return arma::zeros<arma::vec>(size);
   }

    if ((amp.n_elem == size) && (phase.n_elem == size)) {
        if (!m_dynamicCorrStarted) {
            m_dynamicCorrStarted = true;
           Logger::Logger() << "Dynamic correction started.";
        }

        int ntap = FIR_NTAP;
        double fs = 150;
        double f = 10;

        arma::vec time = arma::linspace<arma::vec>(0, ntap-1, ntap);
        arma::mat t_mat = arma::repmat(time.t(), size, 1)/fs;

        arma::mat phase_mat = arma::repmat(phase, 1, ntap) - phref;
        arma::mat fir = arma::cos(2*M_PI*f*t_mat - phase_mat) * 2/fs; // - or + the phase ???

        // Pop front element, them pushback new one
        for (int i = 0 ; i < FIR_NTAP - 1 ; i++) {
             m_values10Hz(i) = m_values10Hz(i+1);
        }
        m_values10Hz(FIR_NTAP-1) = value10Hz;

        arma::vec dynamicCorr = arma::zeros<arma::vec>(size);

        for (int i = 0 ; i < size ; i++){
            dynamicCorr.row(i) = fir.row(i) * m_values10Hz;
        }

        dynamicCorr %= amp/ampref;
        std::cout << arma::max(arma::abs(dynamicCorr)) << '\n'<<'\n';

        return dynamicCorr;
    } else {
        m_dynamicCorrStarted = false;
        Logger::Logger() << "Dynamic correction stopped.";
    }

    return arma::zeros<arma::vec>(size);
}

int CorrectionProcessor::checkRMS(const arma::vec& diffX, const arma::vec& diffY)
{
    double rmsX = (diffX.n_elem-1) * arma::stddev(diffX) / diffX.n_elem;
    double rmsY = (diffY.n_elem-1) * arma::stddev(diffY) / diffY.n_elem;
    if ((rmsX > m_lastRMS.x*1.1) || (rmsY > m_lastRMS.y*1.1))
    {
        m_rmsErrorCnt++;
    //    Logger::Logger() << "RMS error - Nb " << m_rmsErrorCnt << " out of 5.";
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
