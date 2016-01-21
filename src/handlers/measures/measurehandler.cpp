#include "handlers/measures/measurehandler.h"

#include "define.h"
#include "adc.h"
#include "dac.h"
#include "dma.h"
#include <math.h>

MeasureHandler::MeasureHandler(RFMDriver *driver, DMA *dma, bool weightedCorr)
    : Handler(driver, dma, weightedCorr)
{
}

int MeasureHandler::make()
{
    arma::vec diffX, diffY;
    arma::vec CMx, CMy;
    bool newInjection;
    this->getNewData(diffX, diffY, newInjection);
    
    /// SAVE diffX diffY and get amp
    
    
    RFM2G_UINT32 *DACout;
    double CM = sin(2*M_PI * m_f * 150e-3*m_sample);
    DACout[m_CMidx] = CM;
    if (!READONLY) {
        this->writeCorrectors(DACout);
    }
    
    if (m_CMidx == m_nbCM-1 && m_sample == m_maxSample-1 &&  m_f == m_fmax-1){
        std::cout << "done" << std::endl;
        exit(0);
    }
    
    if (m_sample < m_maxSample) {
        m_sample++;
    } else { // m_sample == m_maxSample
        if (m_f < m_fmax) {
            m_f++; // mouveau m_f
            m_sample = 0;
        } else { // m_f == m_fmax => change CM
            m_f = 0; // mouveau m_f
            m_sample = 0;
            m_CMidx++;
        }
    }
}

void MeasureHandler::setProcessor(arma::mat SmatX, arma::mat SmatY,
                                     double IvecX, double IvecY,
                                     double Frequency, 
                                     double P, double I, double D,
                                     arma::vec CMx, arma::vec CMy,
                                     bool weightedCorr)
{
}


