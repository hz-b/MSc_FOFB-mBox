#include "handlers/correction/correctionhandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"

CorrectionHandler::CorrectionHandler(RFMDriver *driver, DMA *dma, bool weightedCorr)
    : Handler(driver, dma, weightedCorr)
{
}

int CorrectionHandler::make()
{
    arma::vec diffX, diffY;
    arma::vec CMx, CMy;
    bool newInjection;
    this->getNewData(diffX, diffY, newInjection);
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
        m_loopDir *= -1; // FIXME: Do we really want this?
        return errornr;
    }

    RFM2G_UINT32   DACout[DAC_BUFFER_SIZE];
    RFM2G_UINT32   DACout2[DAC_BUFFER_SIZE];

    if ((typeCorr & Correction::Horizontal) == Correction::Horizontal) {
        CMx = (CMx % m_scaleDigitsX) + numbers::halfDigits;
        for (int i = 0; i <  m_correctionProcessor.numCMx(); i++)
        {
            int corPos = m_dac->waveIndexYAt(i)-1;
            if (rfm2gMemNumber % 2 == 0) {
                DACout2[corPos] = CMx(i);
            } else {
                DACout[corPos] = CMx(i);
            }
        }
    }
    if ((typeCorr & Correction::Vertical) == Correction::Vertical) {
        CMy = (CMy % m_scaleDigitsY) + numbers::halfDigits;

        for (int i = 0; i < m_correctionProcessor.numCMy(); i++) {
            int corPos = m_dac->waveIndexYAt(i)-1;
            if (rfm2gMemNumber % 2 == 0) {
                DACout2[corPos] = CMy(i);
            } else {
                DACout[corPos] = CMy(i);
            }
        }
    }
    DACout[112] = (m_loopDir*2500000) + numbers::halfDigits;
    DACout[113] = (m_loopDir* (-1) * 2500000) + numbers::halfDigits;
    DACout[114] = (m_loopDir*2500000) + numbers::halfDigits;

    m_loopDir *= -1;
    m_correctionProcessor.checkCorrection();

    if (!READONLY) {
        this->writeCorrectors(DACout);
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
    m_correctionProcessor.setSmat(SmatX, SmatY, IvecX, IvecY, weightedCorr);
    m_correctionProcessor.setInjectionCnt(Frequency);
    m_correctionProcessor.setPID(P,I,D);
    m_correctionProcessor.setCMs(CMx, CMy);
}


