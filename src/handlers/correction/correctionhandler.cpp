#include "handlers/correction/correctionhandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "modules/zmq/logger.h"

CorrectionHandler::CorrectionHandler(RFMDriver *driver, DMA *dma, bool weightedCorr)
    : Handler(driver, dma, weightedCorr)
{
}

int CorrectionHandler::typeCorrection()
{
    int typeCorr = Correction::None;
    if ((m_plane == 0) || (m_plane == 1) || ((m_plane == 3) && (m_loopDir > 0))) {
        typeCorr |= Correction::Horizontal ;
    }
    if ((m_plane == 0) || (m_plane == 2) || ((m_plane == 3) && (m_loopDir < 0))) {
        typeCorr |= Correction::Vertical;
    }
    return typeCorr;
}

int CorrectionHandler::callProcessorRoutine(const CorrectionInput_t& input,
                                            arma::vec& CMx, arma::vec& CMy)
{
    int correctionError = m_correctionProcessor.process(input, CMx, CMy);
    if (correctionError) {
        return correctionError;
    }

    // If this has an error, we don't care: it's not deadly and we have no way
    // to  handle it.
    m_dyn10HzCorrectionProcessor.process(input, CMx, CMy);

    return 0;
}

void CorrectionHandler::setProcessor(arma::mat SmatX, arma::mat SmatY,
                                     double IvecX, double IvecY,
                                     double Frequency,
                                     double P, double I, double D,
                                     arma::vec CMx, arma::vec CMy,
                                     bool weightedCorr)
{
    m_correctionProcessor.initCMs(CMx, CMy);
    m_correctionProcessor.initSmat(SmatX, SmatY, IvecX, IvecY, weightedCorr);
    m_correctionProcessor.initInjectionCnt(Frequency);
    m_correctionProcessor.initPID(P,I,D);
    m_correctionProcessor.finishInitialization();

    m_dyn10HzCorrectionProcessor.initialize();
}


