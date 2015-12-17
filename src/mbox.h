#ifndef MBOX_H
#define MBOX_H

#include <armadillo>
#include "define.h"

namespace numbers {
    const double cf = 0.3051758e-3;
    const double halfDigits   = 1<<23;
}

class CorrectionHandler;
class RFMDriver;
class RFMHelper;
class ADC;
class DAC;
class DMA;

enum RunStatus
{
    Idle = 0,
    Running = 1,
};

enum RunState
{
    Preinit = 0,
    Initialized = 1,
    Error = 2
};

class mBox
{
public:
    explicit mBox(char *deviceName, bool weightedCorr, bool readOnly);
    ~mBox();
    void startLoop();
    void sendMessage(const char* message, const char *error);


private:
    void initValues();
    void initRFM(char *deviceName);
    void initIndexes(double *ADC_WaveIndexX);
    void getNewData(arma::vec &diffX, arma::vec &diffY, bool &newInjection);
    void doCorrection();
    void writeCorrection(RFM2G_UINT32* DACout);
    int getIdx(char numBPMs, double * ADC_BPMIndex_Pos, double DeviceWaveIndex);
    void postError(unsigned int errornr);

    bool m_readOnly;
    RunState m_runningState;
    RunStatus m_runningStatus;
    ADC *m_adc;
    DAC *m_dac;
    DMA *m_dma;
    CorrectionHandler *m_corrHandler;
    RFMDriver *m_driver;
    RFMHelper *m_rfmHelper;
    bool m_weightedCorr;
    //FS BUMP
    int m_idxHBP2D6R,
	m_idxBPMZ6D6R,
	m_idxHBP1D5R,
	m_idxBPMZ3D5R,
	m_idxBPMZ4D5R,
	m_idxBPMZ5D5R,
	m_idxBPMZ6D5R;
    double m_loopDir;
    double m_plane;
    arma::vec m_scaleDigitsX, m_scaleDigitsY;

    arma::vec m_gainX, m_gainY;
    int m_numBPMx, m_numBPMy;
    arma::vec m_BPMoffsetX, m_BPMoffsetY;
};

#endif // MBOX_H
