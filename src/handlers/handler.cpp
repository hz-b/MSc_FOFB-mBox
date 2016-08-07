/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "handlers/handler.h"

#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "rfm_helper.h"
#include "modules/timers.h"
#include "modules/zmq/logger.h"
#include "modules/zmq/messenger.h"

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

    std::vector<double> ADC_WaveIndexX;
    std::vector<double> ADC_WaveIndexY;
    std::vector<double> DAC_WaveIndexX;
    std::vector<double> DAC_WaveIndexY;

    arma::mat SmatX;
    arma::mat SmatY;

    arma::vec CMx;
    arma::vec CMy;

    double IvecX, IvecY;
    double Frequency;
    double P, I, D;

    RFMHelper rfmHelper(m_driver, m_dma);
    // ADC/DAC
    rfmHelper.readStruct("ADC_BPMIndex_PosX", ADC_WaveIndexX, RFMHelper::readStructtype_pchar);
    rfmHelper.readStruct("ADC_BPMIndex_PosY", ADC_WaveIndexY, RFMHelper::readStructtype_pchar);
    rfmHelper.readStruct("DAC_HCMIndex", DAC_WaveIndexX, RFMHelper::readStructtype_pchar);
    rfmHelper.readStruct("DAC_VCMIndex", DAC_WaveIndexY, RFMHelper::readStructtype_pchar);
    // Smatrix
    rfmHelper.readStruct("SmatX", SmatX, RFMHelper::readStructtype_mat);
    rfmHelper.readStruct("SmatY", SmatY, RFMHelper::readStructtype_mat);
    // Parameters
    rfmHelper.readStruct("GainX", m_gain.x, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("GainY", m_gain.y, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("BPMoffsetX", m_BPMoffset.x, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("BPMoffsetY", m_BPMoffset.y, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("scaleDigitsH", m_scaleDigits.x, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("scaleDigitsV", m_scaleDigits.y, RFMHelper::readStructtype_vec);
    // Correctors
    rfmHelper.readStruct("P", P, RFMHelper::readStructtype_double);
    rfmHelper.readStruct("I", I, RFMHelper::readStructtype_double);
    rfmHelper.readStruct("D", D, RFMHelper::readStructtype_double);
    rfmHelper.readStruct("plane", m_plane, RFMHelper::readStructtype_double);
    rfmHelper.readStruct("Frequency", Frequency, RFMHelper::readStructtype_double);
    // Singular Values
    rfmHelper.readStruct( "SingularValueX", IvecX, RFMHelper::readStructtype_double);
    rfmHelper.readStruct( "SingularValueY", IvecY, RFMHelper::readStructtype_double);
    // CM
    rfmHelper.readStruct("CMx", CMx, RFMHelper::readStructtype_vec);
    rfmHelper.readStruct("CMy", CMy, RFMHelper::readStructtype_vec);

 //   rfmHelper.readStruct("DACout", m_DACout, readStructtype_pchar);
    m_numBPM.x = SmatX.n_rows;
    m_numBPM.y = SmatY.n_rows;
    m_numCM.x = SmatX.n_cols;
    m_numCM.y = SmatY.n_cols;

    m_dac->setWaveIndexX(DAC_WaveIndexX);
    m_dac->setWaveIndexY(DAC_WaveIndexY);
    m_adc->setWaveIndexX(ADC_WaveIndexX);
    m_adc->setWaveIndexY(ADC_WaveIndexY);

    this->setProcessor(SmatX, SmatY, IvecX, IvecY, Frequency, P/100, I/100, D/100, CMx, CMy, m_weightedCorr);

    this->initIndexes(ADC_WaveIndexX);

    Messenger::updateMap("SMAT-X", SmatX);
    Messenger::updateMap("SMAT-Y", SmatY);
    Messenger::updateMap("IVEC-X", IvecX);
    Messenger::updateMap("IVEC-Y", IvecY);
    Messenger::updateMap("P", P);
    Messenger::updateMap("I", I);
    Messenger::updateMap("D", D);
    Messenger::updateMap("BPM-OFFSET-X", m_BPMoffset.x);
    Messenger::updateMap("BPM-OFFSET-Y", m_BPMoffset.y);
    Messenger::updateMap("FREQUENCY", Frequency);
    Messenger::updateMap("NB-BPM-X", m_numBPM.x);
    Messenger::updateMap("NB-BPM-Y", m_numBPM.y);
    Messenger::updateMap("NB-CM-X", m_numCM.x);
    Messenger::updateMap("NB-CM-Y", m_numCM.y);
    Messenger::updateMap("CM-X", CMx);
    Messenger::updateMap("CM-Y", CMy);

    if (!READONLY) {
        m_adc->init();
        m_dac->changeStatus(DAC_ENABLE);
    }
}

void Handler::initIndexes(const std::vector<double>& ADC_WaveIndexX)
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

int Handler::getIdx(const std::vector<double>& ADC_BPMIndex_Pos, double DeviceWaveIndex)
{
    auto it = std::find(ADC_BPMIndex_Pos.begin(), ADC_BPMIndex_Pos.end(), DeviceWaveIndex);
    int position = it - ADC_BPMIndex_Pos.begin();

    return position;
}

int Handler::make()
{
    TimingModule::addTimer(_ME_);

    arma::vec diffX, diffY;
    arma::vec CMx = arma::zeros<arma::vec>(m_numCM.x);
    arma::vec CMy = arma::zeros<arma::vec>(m_numCM.y);;
    bool newInjection = false;

    TimingModule::addTimer("ADC_Full");
    int readError = this->getNewData(diffX, diffY, newInjection);
    if (readError)
    {
        Logger::error(_ME_) << "Cannot correct, error in data acquisition";
        return readError;
    }
    TimingModule::timer("ADC_Full").stop();

    Logger::values(LogValue::BPM, m_dma->status()->loopPos, std::vector<arma::vec>({diffX, diffY}));
    Logger::values(LogValue::ADC, m_dma->status()->loopPos, std::vector<std::vector<RFM2G_INT16> >({m_adc->buffer()}));

    CorrectionInput_t input;

    input.typeCorr = this->typeCorrection();

    input.diff.x = diffX;
    input.diff.y = diffY;
    input.newInjection = newInjection;
    input.value10Hz = m_adc->bufferAt(62);

    TimingModule::addTimer("Computation");
    int errornr = this->callProcessorRoutine(input, CMx, CMy);
    TimingModule::timer("Computation").stop();
    if (errornr) {
        return errornr;
    }

    Logger::values(LogValue::CM, m_dma->status()->loopPos, std::vector<arma::vec>({CMx, CMy}));

    TimingModule::addTimer("DAC_Full");
    this->prepareCorrectionValues(CMx, CMy, input.typeCorr);

    if (!READONLY) {
        int writeError = this->writeCorrection();
        if (writeError) {
            return writeError;
        }
    }
    TimingModule::timer("DAC_Full").stop();

    TimingModule::timer(_ME_).stop();

    return 0;
}

int Handler::getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection)
{
    arma::vec rADCdataX(m_numBPM.x), rADCdataY(m_numBPM.y);
    if (m_adc->read()) {
        Logger::error(_ME_) << "Read Error";
        return Error::ADC;
    }

    for (unsigned int i = 0; i < m_numBPM.x; i++) {
        unsigned int  lADCPos = m_adc->waveIndexXAt(i)-1;
        rADCdataX(i) =  m_adc->bufferAt(lADCPos);
    }

    for (unsigned int i = 0; i < m_numBPM.y; i++) {
        unsigned int lADCPos = m_adc->waveIndexYAt(i)-1;
        rADCdataY(i) =  m_adc->bufferAt(lADCPos);
    }

    newInjection = (m_adc->bufferAt(INJECT_TRIG) > 1000);

    diffX = (rADCdataX % m_gain.x * numbers::cf * -1 ) - m_BPMoffset.x;
    diffY = (rADCdataY % m_gain.y * numbers::cf      ) - m_BPMoffset.y;
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
        arma::vec Data_CMx = (CMx % m_scaleDigits.x) + numbers::halfDigits;
        for (int i = 0; i <  Data_CMx.n_elem; i++)
        {
            int corPos = m_dac->waveIndexXAt(i)-1;
            m_DACout[corPos] = Data_CMx(i);
        }
    }
    if ((typeCorr & Correction::Vertical) == Correction::Vertical) {
        arma::vec Data_CMy = (CMy % m_scaleDigits.y) + numbers::halfDigits;

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


int Handler::writeCorrection()
{
    if (m_dac->write(m_plane, m_loopDir, m_DACout) > 0) {
         return Error::DAC;
    }
    unsigned long pos = STATUS_MEMPOS;

    return 0;
}
