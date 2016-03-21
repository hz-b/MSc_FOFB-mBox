#include "handlers/correction/correctionhandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "logger/logger.h"

CorrectionHandler::CorrectionHandler(RFMDriver *driver, DMA *dma, bool weightedCorr)
    : Handler(driver, dma, weightedCorr)
{
}

int CorrectionHandler::make()
{
    arma::vec diffX, diffY;
    arma::vec CMx, CMy;
    bool newInjection;
    if (this->getNewData(diffX, diffY, newInjection))
    {
        Logger::error(_ME_) << "Cannot correct, error in data acquisition" << Logger::flush;
        return 1;
    }
    RFM2G_UINT32 rfm2gMemNumber = m_dma->status()->loopPos;

    int typeCorr = Correction::None;
    if ((m_plane == 0) || (m_plane == 1) || ((m_plane == 3) && (m_loopDir > 0))) {
        typeCorr |= Correction::Horizontal ;
    }
    if ((m_plane == 0) || (m_plane == 2) || ((m_plane == 3) && (m_loopDir < 0))) {
        typeCorr |= Correction::Vertical;
    }
    int errornr = m_correctionProcessor.correct(diffX, diffY,
                                                newInjection,
                                                CMx, CMy,
                                                typeCorr);
    if (errornr) {
        return errornr;
    }

    this->prepareCorrectionValues(CMx, CMy, typeCorr);

    if (!READONLY) {
        this->writeCorrection();
    }
    return 0;
}

void CorrectionHandler::setProcessor(arma::mat SmatX, arma::mat SmatY,
                                     double IvecX, double IvecY,
                                     double Frequency,
                                     double P, double I, double D,
                                     arma::vec CMx, arma::vec CMy,
                                     bool weightedCorr)
{
    m_correctionProcessor.setCMs(CMx, CMy);
    m_correctionProcessor.setSmat(SmatX, SmatY, IvecX, IvecY, weightedCorr);
    m_correctionProcessor.setInjectionCnt(Frequency);
    m_correctionProcessor.setPID(P,I,D);
}


