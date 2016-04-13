#include "handlers/handler.h"

#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "rfm_helper.h"
#include "logger/logger.h"
#include "logger/messenger.h"

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
    Logger::Logger() << "Disable handler";
    m_adc->stop();
    m_dac->changeStatus(DAC_DISABLE);
}

void Handler::init()
{
    Logger::Logger() << "Read Data from RFM";

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
    rfmHelper.readStruct("GainX", m_gainX, readStructtype_vec);
    rfmHelper.readStruct("GainY", m_gainY, readStructtype_vec);
    rfmHelper.readStruct("BPMoffsetX", m_BPMoffsetX, readStructtype_vec);
    rfmHelper.readStruct("BPMoffsetY", m_BPMoffsetY, readStructtype_vec);
    rfmHelper.readStruct("scaleDigitsH", m_scaleDigitsX, readStructtype_vec);
    rfmHelper.readStruct("scaleDigitsV", m_scaleDigitsY, readStructtype_vec);
    // Correctors
    rfmHelper.readStruct("P", P, readStructtype_double);
    rfmHelper.readStruct("I", I, readStructtype_double);
    rfmHelper.readStruct("D", D, readStructtype_double);
    rfmHelper.readStruct("plane", m_plane, readStructtype_double);
    rfmHelper.readStruct("Frequency", Frequency, readStructtype_double);
    // Singular Values
    rfmHelper.readStruct( "SingularValueX", IvecX, readStructtype_double);
    rfmHelper.readStruct( "SingularValueY", IvecY, readStructtype_double);
    // CM
    rfmHelper.readStruct("CMx", CMx, readStructtype_vec);
    rfmHelper.readStruct("CMy", CMy, readStructtype_vec);

 //   rfmHelper.readStruct("DACout", m_DACout, readStructtype_pchar);
    m_numBPMx = SmatX.n_rows;
    m_numBPMy = SmatY.n_rows;
    m_numCMx = SmatX.n_cols;
    m_numCMy = SmatY.n_cols;

    m_dac->setWaveIndexX(std::vector<double>(DAC_WaveIndexX, DAC_WaveIndexX+128));
    m_dac->setWaveIndexY(std::vector<double>(DAC_WaveIndexY, DAC_WaveIndexY+128));
    m_adc->setWaveIndexX(std::vector<double>(ADC_WaveIndexX, ADC_WaveIndexX+128));
    m_adc->setWaveIndexY(std::vector<double>(ADC_WaveIndexY, ADC_WaveIndexY+128));

    this->setProcessor(SmatX, SmatY, IvecX, IvecY, Frequency, P/100, I/100, D/100, CMx, CMy, m_weightedCorr);

    this->initIndexes(std::vector<double>(ADC_WaveIndexX, ADC_WaveIndexX+128));

    Messenger::updateMap("SMAT-X", SmatX);
    Messenger::updateMap("SMAT-Y", SmatY);
    Messenger::updateMap("IVEC-X", IvecX);
    Messenger::updateMap("IVEC-Y", IvecY);
    Messenger::updateMap("P", P);
    Messenger::updateMap("I", I);
    Messenger::updateMap("D", D);
    Messenger::updateMap("BPM-OFFSET-X", m_BPMoffsetX);
    Messenger::updateMap("BPM-OFFSET-Y", m_BPMoffsetY);
    Messenger::updateMap("FREQUENCY", Frequency);
    Messenger::updateMap("NB-BPM-X", m_numBPMx);
    Messenger::updateMap("NB-BPM-Y", m_numBPMy);
    Messenger::updateMap("NB-CM-X", m_numCMx);
    Messenger::updateMap("NB-CM-Y", m_numCMy);

    if (!READONLY) {
        m_adc->init();
        m_dac->changeStatus(DAC_ENABLE);
    }
}

void Handler::initIndexes(const std::vector<double> &ADC_WaveIndexX)
{
    Logger::Logger() << "Init Indexes";
    //FS BUMP
    m_idxHBP2D6R  = 160; //(2*81)-1(X) -1(C)
    m_idxBPMZ6D6R = getIdx(ADC_WaveIndexX, 163);
    Logger::Logger() << "\tidx 6D6 : " << m_idxBPMZ6D6R;
    //ARTOF
    m_idxHBP1D5R  = 142; //(2*72)-1(x) -1(C)
    m_idxBPMZ3D5R = getIdx(ADC_WaveIndexX, 123);
    Logger::Logger() << "\tidx 3Z5 : " << m_idxBPMZ3D5R;
    m_idxBPMZ4D5R = getIdx(ADC_WaveIndexX, 125);
    Logger::Logger() << "\tidx 4Z5 : " << m_idxBPMZ4D5R;
    m_idxBPMZ5D5R = getIdx(ADC_WaveIndexX, 129);
    Logger::Logger() << "\tidx 5Z5 : " << m_idxBPMZ5D5R;
    m_idxBPMZ6D5R = getIdx(ADC_WaveIndexX, 131);
    Logger::Logger() << "\tidx 6Z5 : " << m_idxBPMZ6D5R;
}

int Handler::getIdx(const std::vector<double> &ADC_BPMIndex_Pos, double DeviceWaveIndex)
{
    for (int i = 0; i < ADC_BPMIndex_Pos.size(); i++) {
        if (ADC_BPMIndex_Pos.at(i) == DeviceWaveIndex)
            return i;
    }
    return ADC_BPMIndex_Pos.size();
}

int Handler::make()
{
    arma::vec diffX, diffY;
    arma::vec CMx, CMy;
    bool newInjection = false;
    if (this->getNewData(diffX, diffY, newInjection))
    {
        Logger::error(_ME_) << "Cannot correct, error in data acquisition";
        return 1;
    }
    Logger::values(LogValue::BPM, m_dma->status()->loopPos, diffX, diffY);

    int typeCorr = this->typeCorrection();
    int errornr = this->callProcessorRoutine(diffX, diffY,
                                             newInjection,
                                             CMx, CMy,
                                             typeCorr);
    if (errornr) {
        return errornr;
    }

    Logger::values(LogValue::CM, m_dma->status()->loopPos, CMx, CMy);
    this->prepareCorrectionValues(CMx, CMy, typeCorr);

    if (!READONLY) {
        this->writeCorrection();
    }
    return 0;
}

int Handler::getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection)
{
    arma::vec rADCdataX(m_numBPMx), rADCdataY(m_numBPMy);
    if (m_adc->read()) {
        Logger::postError(FOFB_ERROR_ADC);
        Logger::error(_ME_) << "Read Error";
        return FOFB_ERROR_ADC;
    } 

    for (unsigned int i = 0; i < m_numBPMx; i++) {
        unsigned int  lADCPos = m_adc->waveIndexXAt(i)-1;
        rADCdataX(i) =  m_adc->bufferAt(lADCPos);
    }
    
    for (unsigned int i = 0; i < m_numBPMy; i++) {
        unsigned int lADCPos = m_adc->waveIndexYAt(i)-1;
        rADCdataY(i) =  m_adc->bufferAt(lADCPos);
    }
    
    
    newInjection = (m_adc->bufferAt(110) > 1000); 

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

    return 0;
}

void Handler::prepareCorrectionValues(const arma::vec& CMx, const arma::vec& CMy, int typeCorr)
{
    if ((typeCorr & Correction::Horizontal) == Correction::Horizontal) {
        arma::vec Data_CMx = (CMx % m_scaleDigitsX) + numbers::halfDigits;
        for (int i = 0; i <  Data_CMx.n_elem; i++)
        {
            int corPos = m_dac->waveIndexXAt(i)-1;
            m_DACout[corPos] = Data_CMx(i);
        }
    }
    if ((typeCorr & Correction::Vertical) == Correction::Vertical) {
        arma::vec Data_CMy = (CMy % m_scaleDigitsY) + numbers::halfDigits;

        for (int i = 0; i < Data_CMy.n_elem; i++) {
            int corPos = m_dac->waveIndexYAt(i)-1;
            m_DACout[corPos] = Data_CMy(i);
        }
    }
    m_DACout[112] = (m_loopDir*2500000) + numbers::halfDigits;
    m_DACout[113] = (m_loopDir* (-1) * 2500000) + numbers::halfDigits;
    m_DACout[114] = (m_loopDir*2500000) + numbers::halfDigits;

    m_loopDir *= -1;
}


void Handler::writeCorrection()
{
    if (m_dac->write(m_plane, m_loopDir, m_DACout) > 0) {
         m_dma->status()->errornr = FOFB_ERROR_DAC;
         Logger::postError(FOFB_ERROR_DAC);
    }
    unsigned long pos = STATUS_MEMPOS;
    m_driver->write(pos, m_dma->status(), sizeof(t_status));
}
