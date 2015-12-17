#include "mbox.h"

#include <time.h>
#include <unistd.h>
#include <iostream>

#include "adc.h"
#include "correctionhandler.h"
#include "dac.h"
#include "dma.h"
#include "rfmdriver.h"
#include "rfm_helper.h"

mBox::mBox(char *deviceName, bool weightedCorr, bool readOnly)
{
    m_weightedCorr = weightedCorr;
    m_runningState = Preinit;
    m_runningStatus = Idle;
    m_loopDir = 1;
    m_readOnly = readOnly;

    RFM2GHANDLE RFM_handle = 0;
    m_driver = new RFMDriver(RFM_handle);

    this->initRFM( deviceName );

    m_dma = new DMA(m_driver);
    if ( int res = m_dma->init() )
    {
        std::cout << "DMA Error .... Quit" << std::endl;
        exit(res);
    }
    m_rfmHelper = new RFMHelper(m_driver, m_dma);

    m_adc = new ADC(m_driver, m_dma);
    m_dac = new DAC(m_driver, m_dma);
    m_corrHandler = new CorrectionHandler();
}

mBox::~mBox()
{
    delete m_dac,
           m_adc,
           m_rfmHelper, 
           m_dma,
           m_driver,
           m_corrHandler;
}

void mBox::initRFM(char *deviceName)
{
    std::cout << "Init RFM" << std::endl;
    std::cout << "  RFM Handle : " << m_driver->handle() << std::endl;

    if (m_driver->open(deviceName)) {
        std::cout << "  Can't open " << deviceName << std::endl; 
        exit(1); 
    }

    RFM2G_NODE nodeId;
    if (m_driver->nodeId(&nodeId)) {
        std::cout << "  Can't get Node Id" << std::endl;
        exit(1);
    }
    std::cout << "  RFM Node Id : " << nodeId << std::endl;
}

void mBox::startLoop()
{
    for(;;) {
        m_driver->read(CTRL_MEMPOS, &m_runningStatus, 1);
        m_runningStatus = Running; // HACK
        if (m_runningStatus == 33) {
            std::cout << "  !!! MDIZ4T4R was restarted !!! ... Wait for initialization " << std::endl;
            while (m_runningStatus != Idle) {
                m_driver->read(CTRL_MEMPOS , &m_runningStatus , 1);
                sleep(1);
            }
            std::cout << "Wait for start" << std::endl;
        }

        // if Idle, don't do anything
        if ((m_runningStatus == Idle) && (m_runningState == Preinit)) {}

        /**
         * Initialize correction
         */
        if ((m_runningStatus == Running) && (m_runningState == Preinit)) {
            this->initValues(); 
            m_runningState = Initialized;
            if (!m_readOnly) {
                m_dac->changeStatus(DAC::Start);
            }

            std::cout << "RUN RUN RUN .... " << std::endl << std::flush;
        }

        /**
         * Read and correct
         */
        if ((m_runningStatus == Running) && (m_runningState == Initialized)) {
            this->doCorrection();
        }

        /**
         * Stop correction
         */
        if ((m_runningStatus == Idle) && (m_runningState != Preinit)) {
            std::cout << "Stopped  ....." << std::endl << std::flush;
            m_runningState = Preinit;
        }

        struct timespec t_stop;
        t_stop.tv_sec=0;
        t_stop.tv_nsec=1000000;
        clock_nanosleep(CLOCK_MODE, 0, &t_stop, 0);
    }
}

void mBox::getNewData(vec &diffX, vec &diffY, bool &newInjection)
{
    vec rADCdataX(m_numBPMx), rADCdataY(m_numBPMy);
    if (m_adc->read()) {
         this->postError(FOFB_ERROR_ADC);
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

void mBox::doCorrection()
{
    vec diffX, diffY;
    vec CMx, CMy;
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
    int errornr = m_corrHandler->correct(diffX, diffY, newInjection, CMx, CMy, typeCorr);
    if (errornr) {
        this->postError(errornr);
        m_runningState = Error;
        cout << "error: " << errornr;
    }

    RFM2G_UINT32   DACout[DAC_BUFFER_SIZE];
    RFM2G_UINT32   DACout2[DAC_BUFFER_SIZE];

    if ((typeCorr & Correction::Horizontal) == Correction::Horizontal 
            & !errornr) {
        CMx = (CMx % m_scaleDigitsX) + numbers::halfDigits;
        for (int i = 0; i <  m_corrHandler->numCMx(); i++)
        {
            int corPos = m_dac->waveIndexYAt(i)-1;
            if (rfm2gMemNumber % 2 == 0) {
                DACout2[corPos] = CMx(i);
            } else {
                DACout[corPos] = CMx(i);
            }
        }
    }
    if ((typeCorr & Correction::Vertical) == Correction::Vertical
            & !errornr) {
        CMy = (CMy % m_scaleDigitsY) + numbers::halfDigits;

        for (int i = 0; i < m_corrHandler->numCMy(); i++) {
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
    m_corrHandler->checkCorrection();

    if (!m_readOnly) {
        this->writeCorrection(DACout);
    }
}

void mBox::writeCorrection(RFM2G_UINT32* DACout)
{
    if (m_dac->write(m_plane, m_loopDir, DACout) > 0) {
         m_dma->status()->errornr = FOFB_ERROR_DAC;
         this->postError(FOFB_ERROR_DAC);
    }
    unsigned long pos = STATUS_MEMPOS;
    m_driver->write(pos, m_dma->status(), sizeof(t_status));
}

void mBox::initValues()
{
    std::cout << "Read Data from RFM" << std::endl;

    double ADC_WaveIndexX[128];
    double ADC_WaveIndexY[128];
    double DAC_WaveIndexX[128];
    double DAC_WaveIndexY[128];

    mat SmatX;
    mat SmatY;

    vec CMx;
    vec CMy;

    double IvecX, IvecY; 
    double Frequency;
    double P, I, D;

    // ADC/DAC
    m_rfmHelper->readStruct("ADC_BPMIndex_PosX", ADC_WaveIndexX, readStructtype_pchar);
    m_rfmHelper->readStruct("ADC_BPMIndex_PosY", ADC_WaveIndexY, readStructtype_pchar);
    cout <<  ADC_WaveIndexX[1];
    m_rfmHelper->readStruct("DAC_HCMIndex", DAC_WaveIndexX, readStructtype_pchar);
    m_rfmHelper->readStruct("DAC_VCMIndex", DAC_WaveIndexY, readStructtype_pchar);
    // Smatrix
    m_rfmHelper->readStruct("SmatX", SmatX, readStructtype_mat);
    m_rfmHelper->readStruct("SmatY", SmatY, readStructtype_mat);
    // Parameters
    m_rfmHelper->readStruct( "GainX", m_gainX, readStructtype_vec);
    m_rfmHelper->readStruct( "GainY", m_gainY, readStructtype_vec);
    m_rfmHelper->readStruct( "BPMoffsetX", m_BPMoffsetX, readStructtype_vec);
    m_rfmHelper->readStruct( "BPMoffsetY", m_BPMoffsetY, readStructtype_vec);
    m_rfmHelper->readStruct( "scaleDigitsH", m_scaleDigitsX, readStructtype_vec);
    m_rfmHelper->readStruct( "scaleDigitsV", m_scaleDigitsY, readStructtype_vec);
    // Correctors
    m_rfmHelper->readStruct( "P", P, readStructtype_double);
    m_rfmHelper->readStruct( "I", I, readStructtype_double);
    m_rfmHelper->readStruct( "D", D, readStructtype_double);
    m_rfmHelper->readStruct( "plane", m_plane, readStructtype_double);
    m_rfmHelper->readStruct( "Frequency", Frequency, readStructtype_double);
    // Singular Values	
    m_rfmHelper->readStruct( "SingularValueX", IvecX, readStructtype_double);
    m_rfmHelper->readStruct( "SingularValueY", IvecY, readStructtype_double);
    // CM
    m_rfmHelper->readStruct( "CMx", CMx, readStructtype_vec);
    m_rfmHelper->readStruct( "CMy", CMy, readStructtype_vec);

    m_adc->setWaveIndexX(ADC_WaveIndexX);
    m_adc->setWaveIndexY(ADC_WaveIndexY);

    m_corrHandler->setSmat(SmatX, SmatY, IvecX, IvecY, m_weightedCorr);
    m_corrHandler->setInjectionCnt(Frequency);
    m_corrHandler->setPID(P,I,D);
    m_corrHandler->setCMs(CMx, CMy);

    m_numBPMx = SmatX.n_rows;
    m_numBPMy = SmatY.n_rows;

    this->initIndexes(ADC_WaveIndexX);
}

void mBox::initIndexes(double *ADC_WaveIndexX)
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

int mBox::getIdx(char numBPMs, double* ADC_BPMIndex_Pos, double DeviceWaveIndex)
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

void mBox::postError(unsigned int errornr)
{
    switch (errornr) {
    case 0:
        return;
        break;
    case FOFB_ERROR_ADC:
        this->sendMessage( "FOFB error", "ADC Timeout");
        break;
    case FOFB_ERROR_DAC:
        this->sendMessage( "FOFB error", "DAC Problem");
        break;
    case FOFB_ERROR_CM100:
        this->sendMessage( "FOFB error", "To much to correct");
        break;
    case FOFB_ERROR_NoBeam:
        this->sendMessage( "FOFB error", "No Current");
        break;
    case FOFB_ERROR_RMS:
        this->sendMessage( "FOFB error", "Bad RMS");
        break;
    default:
        this->sendMessage( "FOFB error", "Unknown Problem");
        break;
    }
}

void mBox::sendMessage(const char* message, const char *error)
{
    if (m_readOnly) {
        cout << "Message: " << message;
        if (error)
            cout << " Error: " << error;
        cout << endl;
    } else {
        m_rfmHelper->sendMessage(message, error);
    }
}