#include "handlers/handler.h"

#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "rfm_helper.h"
#include "logger.h"
#include <iostream>

Handler::Handler(RFMDriver *driver, DMA *dma, bool weightedCorr)
{
    m_weightedCorr = weightedCorr;
    m_loopDir = 1;
    m_driver = driver;
    m_dma = dma;
    m_adc = new ADC(m_driver, m_dma);
    m_dac = new DAC(m_driver, m_dma);
}

Handler::~Handler()
{
    delete m_dac,
           m_adc;
}

void Handler::getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection)
{
    arma::vec rADCdataX(m_numBPMx), rADCdataY(m_numBPMy);
    if (m_adc->read()) {
         // this->postError(FOFB_ERROR_ADC);
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
    std::cout << "Read Data from RFM" << std::endl;

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
    std::cout <<  ADC_WaveIndexX[1];
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

    m_adc->setWaveIndexX(ADC_WaveIndexX);
    m_adc->setWaveIndexY(ADC_WaveIndexY);

    this->setProcessor(SmatX, SmatY, IvecX, IvecY, Frequency, P, I, D, CMx, CMy, m_weightedCorr);

    m_numBPMx = SmatX.n_rows;
    m_numBPMy = SmatY.n_rows;

    this->initIndexes(ADC_WaveIndexX);
    
    if (!READONLY) {
        m_dac->changeStatus(DAC::Start);
    }
}

void Handler::initIndexes(double *ADC_WaveIndexX)
{
    //FS BUMP
    m_idxHBP2D6R  = 160; //(2*81)-1(X) -1(C)
    m_idxBPMZ6D6R = getIdx(m_numBPMx, ADC_WaveIndexX, 163);
    std::cout << "idx 6D6 : " << m_idxBPMZ6D6R << std::endl;
    //ARTOF
    m_idxHBP1D5R  = 142; //(2*72)-1(x) -1(C)   
    m_idxBPMZ3D5R = getIdx(m_numBPMx, ADC_WaveIndexX, 123);
    std::cout << "idx 3Z5 : " << m_idxBPMZ3D5R << std::endl;
    m_idxBPMZ4D5R = getIdx(m_numBPMx, ADC_WaveIndexX, 125);
    std::cout << "idx 4Z5 : " << m_idxBPMZ4D5R << std::endl;
    m_idxBPMZ5D5R = getIdx(m_numBPMx, ADC_WaveIndexX, 129);
    std::cout << "idx 5Z5 : " << m_idxBPMZ5D5R << std::endl;
    m_idxBPMZ6D5R = getIdx(m_numBPMx, ADC_WaveIndexX, 131);
    std::cout << "idx 6Z5 : " << m_idxBPMZ6D5R << std::endl;
}

int Handler::getIdx(char numBPMs, double* ADC_BPMIndex_Pos, double DeviceWaveIndex)
{ 
    int res = numBPMs;
    int i;
    //cout << DeviceWaveIndex << endl;
    for (i = 0; i < numBPMs; i++) {
        // cout << ADC_BPMIndex_Pos[i] << " ";
        if (ADC_BPMIndex_Pos[i] == DeviceWaveIndex)
            return i;
    }
    //cout << endl;
    return res;
}

void Handler::writeCorrectors(RFM2G_UINT32* DACout)
{
    if (m_dac->write(m_plane, m_loopDir, DACout) > 0) {
         m_dma->status()->errornr = FOFB_ERROR_DAC;
         Logger::postError(FOFB_ERROR_DAC);
    }
    unsigned long pos = STATUS_MEMPOS;
    m_driver->write(pos, m_dma->status(), sizeof(t_status));
}
