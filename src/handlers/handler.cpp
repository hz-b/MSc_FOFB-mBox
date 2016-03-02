#include "handlers/handler.h"

#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "rfm_helper.h"
#include "logger/logger.h"

#include <iostream>
#include <string>
#include <vector>

Handler::Handler(RFMDriver *driver, DMA *dma, bool weightedCorr)
{
    m_weightedCorr = weightedCorr;
    m_loopDir = 1;
    m_driver = driver;
    m_dma = dma;
    m_adc = new ADC(m_driver, m_dma);
    m_dac = new DAC(m_driver, m_dma);

    if (m_adc->stop()) {
        exit(1);
    }
}


Handler::~Handler()
{
    this->disable();
    delete m_dac,
           m_adc;
}

void Handler::disable()
{
    std::cout << "Disable handler";
    m_adc->stop();
    m_dac->changeStatus(DAC_DISABLE);

}

void Handler::getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection)
{
    arma::vec rADCdataX(m_numBPMx), rADCdataY(m_numBPMy);
    if (m_adc->read()) {
        Logger::postError(FOFB_ERROR_ADC);
	std::cout<< "[Handler::getNewData] ADC::read Error"<< std::endl;
    } else {
         for (unsigned int i = 0; i < m_numBPMx; i++) {
             unsigned int  lADCPos = m_adc->waveIndexXAt(i)-1;
             rADCdataX(i) =  m_adc->bufferAt(lADCPos);
         }

         for (unsigned int i = 0; i < m_numBPMy; i++) {
             unsigned int lADCPos = m_adc->waveIndexYAt(i)-1;
             rADCdataY(i) =  m_adc->bufferAt(lADCPos);
          }

        diffX = (rADCdataX % m_gainX * numbers::cf * -1 ) - m_BPMoffsetX;
        diffY = (rADCdataY % m_gainY * numbers::cf      ) - m_BPMoffsetY;
        //FS BUMP
        double HBP2D6R = m_adc->bufferAt(m_idxHBP2D6R) * numbers::cf * 0.8;
        diffX[m_idxBPMZ6D6R] -= (-0.325 * HBP2D6R);

        //ARTOF
        double HBP1D5R = m_adc->bufferAt(m_idxHBP1D5R) * numbers::cf * 0.8;
        diffX[m_idxBPMZ3D5R] -= (-0.42 * HBP1D5R);
        diffX[m_idxBPMZ4D5R] -= (-0.84 * HBP1D5R);
        diffX[m_idxBPMZ5D5R] -= (+0.84 * HBP1D5R);
        diffX[m_idxBPMZ6D5R] -= (+0.42 * HBP1D5R);
    }
}

void Handler::init()
{
    std::cout << "Read Data from RFM\n";

    double ADC_WaveIndexX[128];
    double ADC_WaveIndexY[128];
    double DAC_WaveIndexX[128];
    double DAC_WaveIndexY[128];

    arma::mat SmatX;
    arma::mat SmatY;

    arma::vec CMx;
    arma::vec CMy;

    double IvecX, IvecY;
    double Frequency;
    double P, I, D;

    RFMHelper rfmHelper(m_driver, m_dma);
    // ADC/DAC
    rfmHelper.readStruct("ADC_BPMIndex_PosX", ADC_WaveIndexX, readStructtype_pchar);
    rfmHelper.readStruct("ADC_BPMIndex_PosY", ADC_WaveIndexY, readStructtype_pchar);

    rfmHelper.readStruct("DAC_HCMIndex", DAC_WaveIndexX, readStructtype_pchar);
    rfmHelper.readStruct("DAC_VCMIndex", DAC_WaveIndexY, readStructtype_pchar);
    // Smatrix
    rfmHelper.readStruct("SmatX", SmatX, readStructtype_mat);
    rfmHelper.readStruct("SmatY", SmatY, readStructtype_mat);
    // Parameters
    rfmHelper.readStruct( "GainX", m_gainX, readStructtype_vec);
    rfmHelper.readStruct( "GainY", m_gainY, readStructtype_vec);
    rfmHelper.readStruct( "BPMoffsetX", m_BPMoffsetX, readStructtype_vec);
    rfmHelper.readStruct( "BPMoffsetY", m_BPMoffsetY, readStructtype_vec);
    rfmHelper.readStruct( "scaleDigitsH", m_scaleDigitsX, readStructtype_vec);
    rfmHelper.readStruct( "scaleDigitsV", m_scaleDigitsY, readStructtype_vec);
    // Correctors
    rfmHelper.readStruct( "P", P, readStructtype_double);
    rfmHelper.readStruct( "I", I, readStructtype_double);
    rfmHelper.readStruct( "D", D, readStructtype_double);
    rfmHelper.readStruct( "plane", m_plane, readStructtype_double);
    rfmHelper.readStruct( "Frequency", Frequency, readStructtype_double);
    // Singular Values
    rfmHelper.readStruct( "SingularValueX", IvecX, readStructtype_double);
    rfmHelper.readStruct( "SingularValueY", IvecY, readStructtype_double);
    // CM
    rfmHelper.readStruct( "CMx", CMx, readStructtype_vec);
    rfmHelper.readStruct( "CMy", CMy, readStructtype_vec);

    m_numBPMx = SmatX.n_rows;
    m_numBPMy = SmatY.n_rows;
    m_numCMx = SmatX.n_cols;
    m_numCMy = SmatY.n_cols;

    m_dac->setWaveIndexX(std::vector<double>(DAC_WaveIndexX, DAC_WaveIndexX+128));
    m_dac->setWaveIndexY(std::vector<double>(DAC_WaveIndexY, DAC_WaveIndexY+128));
    m_adc->setWaveIndexX(std::vector<double>(ADC_WaveIndexX, ADC_WaveIndexX+128));
    m_adc->setWaveIndexY(std::vector<double>(ADC_WaveIndexY, ADC_WaveIndexY+128));
    this->setProcessor(SmatX, SmatY, IvecX, IvecY, Frequency, P, I, D, CMx, CMy, m_weightedCorr);
    this->initIndexes(std::vector<double>(ADC_WaveIndexX, ADC_WaveIndexX+128));

    if (!READONLY) {
        m_adc->init();
        m_dac->changeStatus(DAC_ENABLE);
    }
}

void Handler::initIndexes(const std::vector<double> &ADC_WaveIndexX)
{
    std::cout << "Init Indexes" << std:: endl;
    //FS BUMP
    m_idxHBP2D6R  = 160; //(2*81)-1(X) -1(C)
    m_idxBPMZ6D6R = getIdx(ADC_WaveIndexX, 163);
    std::cout << "\tidx 6D6 : " << m_idxBPMZ6D6R << std::endl;
    //ARTOF
    m_idxHBP1D5R  = 142; //(2*72)-1(x) -1(C)
    m_idxBPMZ3D5R = getIdx(ADC_WaveIndexX, 123);
    std::cout << "\tidx 3Z5 : " << m_idxBPMZ3D5R << std::endl;
    m_idxBPMZ4D5R = getIdx(ADC_WaveIndexX, 125);
    std::cout << "\tidx 4Z5 : " << m_idxBPMZ4D5R << std::endl;
    m_idxBPMZ5D5R = getIdx(ADC_WaveIndexX, 129);
    std::cout << "\tidx 5Z5 : " << m_idxBPMZ5D5R << std::endl;
    m_idxBPMZ6D5R = getIdx(ADC_WaveIndexX, 131);
    std::cout << "\tidx 6Z5 : " << m_idxBPMZ6D5R << std::endl;
}

int Handler::getIdx(const std::vector<double> &ADC_BPMIndex_Pos, double DeviceWaveIndex)
{
    for (int i = 0; i < ADC_BPMIndex_Pos.size() ; i++) {
        if (ADC_BPMIndex_Pos.at(i) == DeviceWaveIndex)
            return i;
    }
    return ADC_BPMIndex_Pos.size();
}

RFM2G_UINT32 *Handler::prepareCorrectionValues(arma::vec& CMx, arma::vec& CMy, int typeCorr)
{
    RFM2G_UINT32 DACout[DAC_BUFFER_SIZE];

    if ((typeCorr & Correction::Horizontal) == Correction::Horizontal) {
        CMx = (CMx % m_scaleDigitsX) + numbers::halfDigits;
        for (int i = 0; i <  CMx.n_elem; i++)
        {
            int corPos = m_dac->waveIndexXAt(i)-1;
            DACout[corPos] = CMx(i);
        }
    }
    if ((typeCorr & Correction::Vertical) == Correction::Vertical) {
        CMy = (CMy % m_scaleDigitsY) + numbers::halfDigits;

        for (int i = 0; i < CMy.n_elem; i++) {
            int corPos = m_dac->waveIndexYAt(i)-1;
            DACout[corPos] = CMy(i);
        }
    }
    DACout[112] = (m_loopDir*2500000) + numbers::halfDigits;
    DACout[113] = (m_loopDir* (-1) * 2500000) + numbers::halfDigits;
    DACout[114] = (m_loopDir*2500000) + numbers::halfDigits;

    m_loopDir *= -1;
}


void Handler::writeCorrection(RFM2G_UINT32* DACout)
{
    std::cout << m_plane << m_loopDir << std::endl;
    if (m_dac->write(m_plane, m_loopDir, DACout) > 0) {
         m_dma->status()->errornr = FOFB_ERROR_DAC;
         Logger::postError(FOFB_ERROR_DAC);
    }
    unsigned long pos = STATUS_MEMPOS;
    m_driver->write(pos, m_dma->status(), sizeof(t_status));
}
