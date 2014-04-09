/***********************************************
Fast mBox
Author: Dennis Engel

************************************************/
//#define ARMA_DONT_USE_BLAS

#include <vector>
#include <armadillo>
#include <iostream>
#include <map>
#include <iomanip>
#include <signal.h>
#include "rfm2g_api.h"

#include <unistd.h>

using namespace arma;
using namespace std;

#define CLOCK_MODE CLOCK_MONOTONIC_RAW

class TimingModule;

class SingleTimer {
  public:
    string name;
    struct timespec t_start;
  public:
    SingleTimer(const string & _name);
    void clock() { 
       clock_gettime(CLOCK_MODE, &t_start);
    }
    void wait(long ns) {
       struct timespec t_stop;
       t_stop.tv_sec=t_start.tv_sec;
       t_stop.tv_nsec=t_start.tv_nsec+ns;
       const long onesec=1000000000;
       if (t_stop.tv_nsec>=onesec) {
          t_stop.tv_sec+=1;
          t_stop.tv_nsec-=onesec;
       }
       clock_nanosleep(CLOCK_MODE,TIMER_ABSTIME,&t_stop,0);
    }
    const timespec getTime() { return t_start; }
    const string & getName() { return name; }
};

template <class T>
T sqr(const T a) { return a*a; }

class TimeDiff {
  string name;
  double t_sum, t_sum2;
  double t_min, t_max;
  long   n_call;
  int    id_start,id_stop;
  TimingModule * timing;
  static const double ms=1.e-3;
 public:
  TimeDiff(const string & _name, int a, int b, TimingModule * tm) { 
     name=_name; 
     id_start=a;
     id_stop=b;
     timing=tm;
     reset();
  }
  void reset() {
    t_sum=t_sum2=0.;
    n_call=0;
    t_min=-1.;
    t_max=-1.;
  }
  void eval();
  double mean() const {
    return t_sum/n_call;
  }
  double mean2() const {
    return t_sum2/n_call;
  }
  double min() const {
    return t_min;
  }
  double max() const {
    return t_max;
  }
  double rms() const {
    return sqrt( mean2() - sqr(mean()) ); 
  }
 friend ostream & operator<<(ostream & s, const TimeDiff & td);
};
  
class TimingModule {
  static TimingModule * theTimingModule;
  map<string,int>      timer_map;
  std::vector<SingleTimer*> all_timer;
  std::vector<TimeDiff*>    all_diffs;

public:
  static TimingModule * getTimingModule() { if (!theTimingModule) theTimingModule=new TimingModule(); return theTimingModule; }
  void addTimer(SingleTimer *);
  void addDiff(const string &name, const string &t_start, const string  &t_stop);
  void eval();  
  void print();
  void reset();
  const string & timerName(const int id) { return all_timer[id]->getName(); }
  double delta(const int id_start, const int id_stop);
  static double delta(const timespec & t_start, const timespec & t_stop);
};

TimingModule * TimingModule::theTimingModule=0;


SingleTimer::SingleTimer(const string & _name) {
  name=_name;
  TimingModule * tm = TimingModule::getTimingModule();
  tm->addTimer(this);
}


void TimeDiff::eval() {
    double dt=timing->delta(id_start,id_stop)/ms;
    t_sum+=dt;
    t_sum2+=dt*dt;
    ++n_call;
    if (dt>t_max) t_max=dt;
    if (dt<t_min || t_min==-1.) t_min=dt;
}

ostream & operator<<(ostream & s, const TimeDiff & td) {
  TimingModule * tm = TimingModule::getTimingModule();
    s<<td.name<<"["<<tm->timerName(td.id_start)<<","<<tm->timerName(td.id_stop)<<"] = "<<td.mean()<<" ("<<td.min()<<","<<td.max()<<")  rms="<<td.rms();
}

 
void TimingModule::addTimer(SingleTimer* timer) {
  size_t id = all_timer.size();
  all_timer.push_back(timer);
  timer_map[timer->getName()]=id;
  cout<<"add Timer "<<timer->getName()<<" "<<id<<endl;
}

void TimingModule::addDiff(const string & name, const string &t_start, const string &t_stop) {
  int a = timer_map[t_start];
  int b = timer_map[t_stop];
  all_diffs.push_back(new TimeDiff(name,a,b,this));
  cout<<"add Diff ("<<a<<","<<b<<")"<<endl;
}

void TimingModule::eval() {
  for (size_t i=0; i<all_diffs.size();++i) all_diffs[i]->eval();
}

void TimingModule::reset() {
  for (size_t i=0; i<all_diffs.size();++i) all_diffs[i]->reset();
}

void TimingModule::print() {
  for (size_t i=0; i<all_diffs.size();++i) { 
    cout<<*(all_diffs[i])<<endl;
  }
}

double TimingModule::delta(const int a, const int b) {
  return delta(all_timer[a]->getTime(),all_timer[b]->getTime());
}

double TimingModule::delta(const timespec & t_start, const timespec & t_stop) {
  return ( t_stop.tv_sec - t_start.tv_sec )
  	+ 1.e-9*( t_stop.tv_nsec - t_start.tv_nsec );
}



SingleTimer t_adc_start("t_adc_start"), t_adc_read("t_adc_read"), t_adc_stop("t_adc_stop");
SingleTimer t_calc_start("t_calc_start"), t_calc_stop("t_calc_stop");
SingleTimer t_dac_start("t_dac_start"), t_dac_clear("t_dac_clear"), 
            t_dac_write("t_dac_write"), t_dac_send("t_dac_send"), t_dac_stop("t_dac_stop");


#define FOFB_ERROR_ADC     1
#define FOFB_ERROR_DAC     2  
#define FOFB_ERROR_CM100   4 
#define FOFB_ERROR_NoBeam  5 
#define FOFB_ERROR_RMS     6  
#define FOFB_ERROR_Unkonwn 7 

char   devicename[] =    "/dev/rfm2g0";
const double cf           = 0.3051758e-3;
const double halfDigits   = 1<<23;
double plane           = 0;
double P, I , D        = 0;
char   result          = 0;
char   numBPMx,numBPMy = 0;
char   numCORx,numCORy = 0;
char   totalnumCORx, totalnumCORy = 0;

struct t_status {
    unsigned short loopPos   ;
    unsigned short errornr   ;
} status ;

struct t_message {
    char * error;
    char * status; 
} message ;

char   Hz = 0;
vec Xsum;
vec Ysum;
vec dCorlastx;
vec dCorlasty;
double IvecX;
double IvecY;

/* DMA Buffer pointer */
volatile char *pDmaMemory;


RFM2GHANDLE    RFM_Handle = 0;
RFM2G_NODE     NodeId;
RFM2GEVENTTYPE ADC_DAC_EVENT = RFM2GEVENT_INTR2;
#define CTRL_MEMPOS       0x03000000

// ADC
#define ADC_BUFFER_SIZE 256
#define ADC_MEMPOS        0x01000000
#define ADC_TIMEOUT       10000   /* milliseconds */
double        *ADC_Data;
RFM2G_INT16    ADC_Buffer[ADC_BUFFER_SIZE];
//RFM2G_INT16    *ADC_Buffer;
char           ADC_Node = 0x01;

// DAC
#define DAC_BUFFER_SIZE 128
#define DAC_MEMPOS        0x02000000
#define DAC_TIMEOUT       60000   /* milliseconds */
RFM2G_UINT32   DACout[DAC_BUFFER_SIZE];
//RFM2G_UINT32  * DACout;

const char *DAC_IOCs[]  = {"IOCS15G","IOCS2G","IOCS4G","IOCS6G","IOCS8G","IOCS10G","IOCS12G","IOCS14G","IOCS16G","IOC3S16G" };
char DAC_nodeIds[]     = {0x02     , 0x12   , 0x14   , 0x16   , 0x18   , 0x1A    , 0x1C    , 0x1E    , 0x20    , 0x21 };
char Act_DAC_nodeIds[] = {0        , 0      , 0      , 1      , 1      , 0       , 0       , 0       , 0       , 0 };
char num_DAC_nodeIds   = 10;



// BPMData
double ADC_WaveIndexX[128];
double ADC_WaveIndexY[128];
vec    diffX, diffY;
vec    GainX, GainY;
vec    BPMoffsetX, BPMoffsetY;
vec    rADCdataX(128);  // Reshape X Data of ADC_Data
vec    rADCdataY(128);  // Reshape Y Data of ADC_Data

//CORData
double DAC_WaveIndexX[128];
double DAC_WaveIndexY[128];
vec CMx;
vec CMy;
mat SmatX;
mat SmatY;
mat SmatInvX;
mat SmatInvY;
vec dCORx, dCORy;
vec scaleDigitsX,scaleDigitsY;
vec dCORxPID, dCORyPID;
vec dCORlastX, dCORlastY;
vec Data_CMx,Data_CMy;

void init_rfm() {
    cout << "Init RFM" << endl;
    result = RFM2gOpen( devicename, &RFM_Handle );
    if (result) {
        cout << "  Can't open " << devicename << "\n" << endl; 
        exit(1); 
    }
    cout << "  RFM Handle : " << RFM_Handle << endl;
    result = RFM2gNodeID (RFM_Handle, &NodeId);
    if (result) {
	cout << "  Can't get Node Id" << endl;
        exit(1);
    }
    cout << "  RFM Node Id : " << NodeId << endl;
}

int init_DMA() {
    #define  DMAOFF_A 0x00100000
    //#define  DMAOFF_A 0xf0000000
    //#define  LINUX_DMA_FLAG 0x0
    #define  LINUX_DMA_FLAG 0x01
    #define  LINUX_DMA_FLAG2 0
    #define  DMA_THRESHOLD 128

    RFM2G_UINT32 rfm2gSize;
    volatile void *pPioCard = NULL;
    volatile char *pDmaCard = NULL;   // alias MappedDmaBase

    cout << "RFM DMA Init " << endl;

    RFM2gSetDMAThreshold(RFM_Handle,DMA_THRESHOLD);

    RFM2GCONFIG rfm2gConfig;
    if(RFM2gGetConfig(RFM_Handle, &rfm2gConfig) == RFM2G_SUCCESS) {
      pPioCard = (char*)rfm2gConfig.PciConfig.rfm2gBase;
      rfm2gSize = rfm2gConfig.PciConfig.rfm2gWindowSize;      
    }

    int pageSize = getpagesize();
    unsigned int numPagesDMA = rfm2gSize / (2* pageSize);
    unsigned int numPagesPIO = rfm2gSize / (2* pageSize);
    if((rfm2gSize % pageSize) > 0) {
       cout << "Increase PIO and DMA " << endl;
        numPagesDMA++;
        numPagesPIO++;
    }

    //numPagesDMA = 100;

    cout << "   pPioCard : " << pPioCard << endl;
    cout << "   rfm2gSize : " << rfm2gSize << endl;
    cout << "   pageSize  : " << pageSize << endl;
    cout << "   numPages DMA/PIO : " << numPagesDMA << endl;

    RFM2G_STATUS rfmReturnStatus = RFM2gUserMemory(RFM_Handle,
                                (volatile void **) (&pDmaCard),
                                (DMAOFF_A | LINUX_DMA_FLAG),
                                numPagesDMA);
    if(rfmReturnStatus != RFM2G_SUCCESS) {
         printf("doDMA: ERROR: Failed to map card DMA buffer; %s\n",
                RFM2gErrorMsg(rfmReturnStatus));
         return -1;
    }
    printf("doDMA: SUCCESS: mapped numPagesDMA=%d at pDmaCard=%p\n", numPagesDMA, pDmaCard);
    pDmaMemory=pDmaCard;


    rfmReturnStatus = RFM2gUserMemoryBytes(RFM_Handle,
                                (volatile void **) (&pPioCard),
                                (0x00000000 | LINUX_DMA_FLAG2),
                                rfm2gSize);
     if(rfmReturnStatus != RFM2G_SUCCESS) {
                printf("doDMA: ERROR: Failed to map card PIO; %s\n",
                        RFM2gErrorMsg(rfmReturnStatus));
                return -1;
     }
     printf("doDMA: Card: PIO memory pointer = 0x%X, Size = 0x%X\n", (int)pPioCard, rfm2gSize);

     return 0;
}

void init_ADC(unsigned int freq, unsigned int DACfreq) {
    cout << "Init ADC" << endl << flush; 
    RFM2G_INT32 ctrl_buffer[128];
    short Navr = DACfreq/freq;
    ctrl_buffer[0] = 512; // RFM2G_LOOP_MAX
    ctrl_buffer[1] = 0;
    ctrl_buffer[2] = 0;
    ctrl_buffer[3] = 2;          // Navr
    //Stop  ADC Sampling (if running)
    if ((freq == 0) || (DACfreq == 0)) {
        cout << "  ADC stop sampling" << endl;
	result = RFM2gSendEvent(RFM_Handle,ADC_Node,ADC_DAC_EVENT,1);
        if (result) { cout << "  Can't stop ADC" << endl; exit(1); }
	cout << "  ADC should be stopped\n" << endl;
	return;
    }
    sleep(2);
    //Write ADC CTRL
    cout << "  ADC write sampling config" << endl;

    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    RFM2gGetDMAThreshold( RFM_Handle, &threshold );

    int data_size = 512;
    if (data_size<threshold) {
         // use PIO transfer
       result = RFM2gWrite( RFM_Handle, 0, &ctrl_buffer,512);
    }
    else {
       RFM2G_INT32 * dst = (RFM2G_INT32*)pDmaMemory; 
       for (int i=0;i<128;++i) {
           dst[i]=ctrl_buffer[i];
       }
       result = RFM2gWrite( RFM_Handle, 0, (void*)pDmaMemory, data_size);
    }
    if (result) { cout << "  Can't write ADC config" << endl; exit(1); }
    //Enable ADC
    cout << "  ADC enable sampling" << endl;
    result = RFM2gSendEvent(RFM_Handle,ADC_Node,ADC_DAC_EVENT,3);
    if (result) { cout << "  Can't enable ADC" << endl; exit(1); }
    //Start Sampling
    cout << "  ADC start sampling" << endl;
    result = RFM2gSendEvent(RFM_Handle,ADC_Node,ADC_DAC_EVENT,2);
    sleep(2);
    if (result) { cout << "  Can't start sampling" << endl; exit(1); }
    cout << "  ADC should be started" << endl;
}


void ctrl_DAC(bool start_stop) {
    char ctrl_date = 0;
    if (start_stop) {
        cout << "starting DACs ... " << endl << flush;
        ctrl_date = 2; //start
    } else {
        cout << "stopping DACs ...." << endl << flush;
        ctrl_date = 1; //stop
    }
    for (int i = 0; i < num_DAC_nodeIds; i++) {
        char nodenr = DAC_nodeIds[i];
	if (Act_DAC_nodeIds[i]) {
	  cout << "  " << DAC_IOCs[i] << "\t: ";
          result = RFM2gSendEvent(RFM_Handle, nodenr, ADC_DAC_EVENT, ctrl_date);
	  if (result) { cout << "Error " << endl; }
	  else        { cout << "successful" << endl; }
        }

    }
}



template <class T>
void prepareField(T& field, unsigned long pos, short dim1, short dim2) {
    RFM2gRead(RFM_Handle,pos,(void*)&field, sizeof(field));

  //  cout << "Got: sizeof: " << sizeof(field) << " [ " << flush ;
 //   for (int i=0; i < dim1*dim2; ++i) {
// 	cout << field[i] << " ";
    //}
    //cout << " ] " << endl;
}

template <>
void prepareField(double *& field, unsigned long pos, short dim1, short dim2) {
    RFM2gRead(RFM_Handle,pos,(void*)&field, dim1*dim2*sizeof(field));

    //cout << "double * Got: [ " << flush ;
    //for (int i=0; i < dim1*dim2; ++i) {
// 	cout << field[i] << " ";
 //   }
   // cout << " ] " << endl;
}



template <>
void prepareField(double& field, unsigned long pos, short dim1, short dim2) {
    RFM2gRead(RFM_Handle,pos,(void*)&field, dim1*dim2*8);
    cout << "d5 Got: " << field << endl;
}

void dumpMemory(void* data, int len) {
    char* p=(char*)data;
    printf("'%f'\n",*(double*)p);
    for (int i=0;i<len;++i) {
       printf("0x%x\n",(int)*(p+i));
    }
}

void dumpMemory(volatile void* data, int len) {
    char* p=(char*)data;
    printf("'%f'\n",*(double*)p);
    for (int i=0;i<len;++i) {
       printf("0x%x\n",(int)*(p+i));
    }
}

template <>
void prepareField(vec& field, unsigned long pos, short dim1, short dim2) {
    RFM2G_UINT32 threshold = 0, data_size=dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    RFM2gGetDMAThreshold( RFM_Handle, &threshold );

    if (data_size<threshold) {
        // use PIO  tranfer
        field.set_size(dim1*dim2);
        RFM2gRead(RFM_Handle,pos,(void*)field.memptr(), data_size);
        //dumpMemory(field.memptr(),8);
    }
    else {
        // use DMA transfer
        RFM2gRead(RFM_Handle,pos,(void*)pDmaMemory, data_size);
        //dumpMemory(pDmaMemory,8);
        field=vec((const double *)pDmaMemory,dim1*dim2);    
    }
    //cout << "v5" << field << endl;
    cout<<"read vec at pos "<<pos<<" len:"<<data_size<<endl;
    //char* p=(char*)field.memptr();
    //printf("'%x'\n",(RFM2G_UINT64)(*(pDmaMemory+pos)));
    //cout<<field<<endl;

}
template <>
void prepareField(mat& field, unsigned long pos, short dim1, short dim2) {
    RFM2G_UINT32 threshold = 0, data_size=dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    RFM2gGetDMAThreshold( RFM_Handle, &threshold );

    if (data_size<threshold) {
         // use PIO transfer
        field.set_size(dim1,dim2);
        RFM2gRead(RFM_Handle,pos,(void*)field.memptr(), dim1*dim2*8);
    }
    else {
        // use DMA transfer
        RFM2gRead(RFM_Handle,pos,(void*)pDmaMemory, data_size);
        //dumpMemory(pDmaMemory,8);
        field=mat((const double *)pDmaMemory,dim1,dim2);    
    }
    //cout << "m5" << field << endl;
    //cout << "Size : " << field.n_cols << ":"<< field.n_rows << endl;
}

template <class T>
void readStruct(const char * structname, T & field, char tartype = 0) {
    unsigned long  pos = CTRL_MEMPOS+1000;
    short elementnr;
    short header[4];
    char  name[80];
    unsigned char  * data;
    
    RFM2gRead(RFM_Handle,pos,&elementnr,2); // 4 * 16-Bit = 8
    //cout << "1 elementnr: " << elementnr << endl;
    pos += 2;
    for(unsigned int i = 0; i<= elementnr; i++) {
        RFM2gRead(RFM_Handle,pos,&header,8); // 4 * 16-Bit = 8
        //cout << "2" << endl;
        pos += 8;
       
        short namesize  = header[0];
        short datasize1 = header[1];
        short datasize2 = header[2];
        short type      = header[3];
        unsigned long  datasize  = datasize1 * datasize2;
        unsigned long  valuenr  = datasize;
        if (type == 1)
                datasize *= 8;               

        RFM2gRead(RFM_Handle,pos,&name,namesize);
	//cout << "3 namesize: " << namesize << endl;

        pos += namesize;
        name[namesize] = 0;
        if (strcmp(structname,name) == 0) {          
	    cout << "   Found Name: " << name << endl;
            prepareField(field,pos,datasize1,datasize2);
            return;
        } 
        pos += datasize;
    }
    cout << "    WARNING : " << structname << " not found !!!" << endl; 
}
#define readStructtype_pchar 0
#define readStructtype_mat 1
#define readStructtype_vec 2
#define readStructtype_double 3
void read_RFMStruct() {   
    cout << "Read Data from RFM" << endl;
    readStruct("ADC_BPMIndex_PosX", ADC_WaveIndexX,readStructtype_pchar);
    readStruct("ADC_BPMIndex_PosY", ADC_WaveIndexY,readStructtype_pchar);
    readStruct("DAC_HCMIndex", DAC_WaveIndexX,readStructtype_pchar);
    readStruct("DAC_VCMIndex", DAC_WaveIndexY,readStructtype_pchar);
    readStruct("SmatX", SmatX,readStructtype_mat);
    readStruct("SmatY", SmatY,readStructtype_mat);
    numBPMx = SmatX.n_rows;
    numCORx = SmatX.n_cols;
    numBPMy = SmatY.n_rows;
    numCORy = SmatY.n_cols;

    diffX      = vec(numBPMx); 
    diffY      = vec(numBPMy);
    readStruct( "GainX" ,GainX,readStructtype_vec);
    readStruct( "GainY", GainY,readStructtype_vec);
    readStruct( "BPMoffsetX", BPMoffsetX,readStructtype_vec);
    readStruct( "BPMoffsetY", BPMoffsetY,readStructtype_vec);
    rADCdataX  = vec(numBPMx);  // Reshape X Data of ADC_Data
    rADCdataY  = vec(numBPMy);  // Reshape Y Data of ADC_Data

    readStruct( "P", P,readStructtype_double);
    readStruct( "I", I,readStructtype_double);
    readStruct( "D", D,readStructtype_double);
    readStruct( "plane", plane,readStructtype_double);
    readStruct( "SingularValueX", IvecX,readStructtype_double);
    readStruct( "SingularValueY", IvecY,readStructtype_double);
    
    //CORData
    readStruct( "CMx", CMx);
    readStruct( "CMy", CMy);
    SmatInvX   = mat(numBPMx,numCORx); 
    SmatInvY   = mat(numBPMy,numCORy); 
    dCORx      = vec(numCORx); 
    dCORy      = vec(numCORy);
    readStruct( "scaleDigitsH", scaleDigitsX,readStructtype_vec);
    readStruct( "scaleDigitsV", scaleDigitsY,readStructtype_vec);

    dCORxPID   = vec(numCORx); 
    dCORyPID   = vec(numCORy);

    dCORlastX  = zeros<vec>(numCORx); 
    dCORlastY  = zeros<vec>(numCORy);
    Xsum       = zeros<vec>(numCORx);
    Ysum       = zeros<vec>(numCORy);
    Data_CMx   = vec(totalnumCORx);
    Data_CMy   = vec(totalnumCORy);
}


void calcSmat() {
    cout << "Calculate Smat" << endl;
    mat U;
    vec s;
    mat S;
    mat V;
    
    cout << "   Given : " << " SmatY cols: " << SmatY.n_cols << " smatY rows " << SmatY.n_rows<< " smatX cols " <<  SmatX.n_cols << " smatX rows" << SmatX.n_rows << "  IvecX : " << IvecX << " IvecY " << IvecY << endl;

    cout << "   SVD Hor" << endl;
    cout << "      make Ivec" << endl;
    if (IvecX > SmatX.n_rows) {
        cout << "IVecX > SmatX.n_rows: Setting IvecX = SmatX.n_rows" << endl;
        IvecX = SmatX.n_rows;
    }
    

   
    cout << "      calc SVD" << endl;
    svd(U,s,V,SmatX);
    cout << "      reduce U to Ivec" << endl;
    U = U.cols(0,IvecX-1);
    cout << "      Transpose U" << endl;
    U = trans(U);
    cout << "      Get Diag Matrix of  S" << endl;
    S = diagmat(s.subvec(0,IvecX-1)); 
    cout << "      reduce V to Ivec" << endl;
    V = V.cols(0,IvecX-1);
    cout << "       Calc new Matrix" << endl;
    SmatInvX = V * inv(S) * U;

    // Vertical Smat
    cout << " SVD Ver" << endl;
    cout << "      make Ivec" << endl;
    if (IvecY > SmatY.n_rows) {
        cout << "IVecY > SmatY.n_rows: Setting IvecY = SmatY.n_rows" << endl;
        IvecY = SmatY.n_rows;
    }
    cout << "      calc SVD" << endl;
    svd(U,s,V,SmatY);
    cout << "      reduce U to Ivec" << endl;
    U = U.cols(0,IvecY-1);
    cout << "      Transpose U" << endl;
    U = trans(U);
    cout << "      Get Diag Matrix of  S" << endl;
    S = diagmat(s.subvec(0,IvecY-1));
    cout << "      reduce V to Ivec" << endl;
    V = V.cols(0,IvecY-1);
    cout << "       Calc new Matrix" << endl;
    SmatInvY = V * inv(S) * U;

    cout << "   SVD complete ..." << endl;
}



unsigned char writeDAC(unsigned int loopPos) {
    RFM2GEVENTINFO EventInfo;           /* Info about received interrupts    */
    RFM2G_UINT32    rfm2gCtrlSeq   = loopPos;
    RFM2G_UINT32    rfm2gMemNumber = loopPos & 0x000ffff;
    // cout << setw(3) << rfm2gMemNumber << " "<< DACout[113] <<"\n";


    /* --- start timer --- */
    t_dac_start.clock();    

    if (RFM2gClearEvent( RFM_Handle, RFM2GEVENT_INTR3)) return 1;
    if (RFM2gEnableEvent( RFM_Handle,  RFM2GEVENT_INTR3)) return 1;
    t_dac_clear.clock();


    /* fill DAC to RFM */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    RFM2gGetDMAThreshold( RFM_Handle, &threshold );


    int data_size = DAC_BUFFER_SIZE*sizeof(RFM2G_UINT32);
    if (data_size<threshold) {
         // use PIO transfer
       if (RFM2gWrite( RFM_Handle, DAC_MEMPOS+(rfm2gMemNumber*data_size),
                          &DACout, data_size)) return 1;
    }
    else {
       RFM2G_INT32 * dst = (RFM2G_INT32*)pDmaMemory; 
       for (int i=0;i<DAC_BUFFER_SIZE;++i) {
           dst[i]=DACout[i];
       }
 
       if (RFM2gWrite( RFM_Handle, DAC_MEMPOS+(rfm2gMemNumber*data_size),
                          (void*)pDmaMemory, data_size)) return 1;
 
    }

    t_dac_write.clock();

    t_adc_start.wait(450000); // sleep 450 us
    //cout << " write "<< t_dac_write.tv_sec << ","<<t_dac_write.tv_nsec <<" \n";

    //   struct timespec t_stop;
    //   t_stop.tv_sec=0;
    //   t_stop.tv_nsec=300000000;
    //clock_nanosleep(CLOCK_MODE,0,&t_stop,0);
     //usleep(300);

    /* tell IOC to work */
    if (RFM2gSendEvent( RFM_Handle, RFM2G_NODE_ALL, RFM2GEVENT_INTR3, 
                            (RFM2G_INT32) rfm2gCtrlSeq)) return 1;
    t_dac_send.clock();    

    /* wait for atleast one ack. */
    EventInfo.Event   = RFM2GEVENT_INTR3;  /* We'll wait on this interrupt */
    EventInfo.Timeout = DAC_TIMEOUT;       /* We'll wait this many milliseconds */
    if (RFM2gWaitForEvent( RFM_Handle, &EventInfo )) return 1;

    // stop timer
    t_dac_stop.clock();    

    return 0;
}

unsigned char readADC() {

    RFM2GEVENTINFO EventInfo;           /* Info about received interrupts    */
    RFM2G_NODE     otherNodeId;         /* Node ID of the other RFM board    */
     /* --- start timer --- */
    //clock_gettime(CLOCK_MODE, &t_adc_start);
    /* Wait on an interrupt from the other Reflective Memory board */
    if (RFM2gClearEvent( RFM_Handle, RFM2GEVENT_INTR1 )) return 1;
    if (RFM2gEnableEvent( RFM_Handle, RFM2GEVENT_INTR1 )) return 1;
     /* --- stop timer --- */
    //clock_gettime(CLOCK_MODE, &t_adc_stop);
    //t_sum_clear+=delta(t_adc_start,t_adc_stop);

   EventInfo.Event    = RFM2GEVENT_INTR1;  /* We'll wait on this interrupt */
    EventInfo.Timeout  = ADC_TIMEOUT;       /* We'll wait this many milliseconds */
    if (RFM2gWaitForEvent( RFM_Handle, &EventInfo )) return 1;
    otherNodeId       = EventInfo.NodeId;
    status.loopPos    = EventInfo.ExtendedInfo;

    /* --- start timer --- */
    t_adc_start.clock();  

    /* Now read data from the other board from BPM_MEMPOS */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    RFM2gGetDMAThreshold( RFM_Handle, &threshold );

    int data_size=ADC_BUFFER_SIZE  *sizeof( RFM2G_INT16 );
    if (data_size<threshold) {
         // use PIO transfer
       if (RFM2gRead( RFM_Handle, ADC_MEMPOS + (status.loopPos * data_size), 
                           (void *)ADC_Buffer, data_size )) return 1;
    }
    else {
       if (RFM2gRead( RFM_Handle, ADC_MEMPOS + (status.loopPos * data_size), 
                           (void *)pDmaMemory, data_size )) return 1;
       RFM2G_INT16 * src = (RFM2G_INT16*)pDmaMemory; 
       for (int i=0;i<ADC_BUFFER_SIZE;++i) {
           ADC_Buffer[i]=src[i];
       }
    }
    /* --- stop timer --- */
    t_adc_read.clock();


    /* Send an interrupt to the other Reflective Memory board */
    if (RFM2gSendEvent( RFM_Handle, otherNodeId, RFM2GEVENT_INTR1, 0)) return 1;
    
    //clock_gettime(CLOCK_MODE, &t_adc_stop);
    //t_sum_send+=delta(t_adc_read,t_adc_stop);
    
    for (char i = 0; i < numBPMx; i++) {
        char lADCPos = ADC_WaveIndexX[i];
        rADCdataX(i) =  ADC_Buffer[lADCPos];
    }

    for (char i = 0; i < numBPMy; i++) {
        char lADCPos = ADC_WaveIndexY[i];
        rADCdataY(i) =  ADC_Buffer[lADCPos];
    }
    
    diffX = (rADCdataX % GainX * cf * -1) - BPMoffsetX;
    diffY = (rADCdataY % GainY * cf * -1) - BPMoffsetY; 

    t_adc_stop.clock();

    return 0;
}



unsigned char make_cor() {

    static int count_test=0;
    static int count=0;
    ++count;
    if (count%15000==0) {
      count_test+=20;
      cout<<"delay="<<count_test<<endl;
      count=0;
    }

    /* start timing */
    t_calc_start.clock();
    //usleep(count_test);
    struct timespec delay;
    delay.tv_sec=0;
    delay.tv_nsec=count_test*1000;

    static double lastrmsX = 999;
    static double lastrmsY = 999;
    static double loopDir  = 1;
    char   rmsErrorCnt = 0;
    //cout << "make cor" << endl;
    //cout << "  prove beam" << endl;
    if (sum(diffX) < -10.5) {
         //cout << " No Beam" << endl;
         //return FOFB_ERROR_NoBeam;
    }
    //cout << "  prove rms" << endl;
    double rmsX = (diffX.n_elem-1) * stddev(diffX) / diffX.n_elem;
    double rmsY = (diffY.n_elem-1) * stddev(diffY) / diffY.n_elem;
    if ((rmsX > lastrmsX*1.1) || (rmsY > lastrmsY*1.1)) {
        rmsErrorCnt++;
        if (rmsErrorCnt > 5) {
            //return FOFB_ERROR_RMS;
        }
    } else {
        rmsErrorCnt = 0;
    }
    lastrmsX = rmsX;
    lastrmsY = rmsY;
    

    //cout << "  calc dCOR" << endl;
    dCORx = SmatInvX * diffX;
    dCORy = SmatInvY * diffY;

    //cout << "  Check dCOR size" << endl;
    //if ((max(dCORx) > 0.100) || (max(dCORy) > 0.100))
    //    return FOFB_ERROR_CM100;


    //cout << "  calc PID" << endl;
    dCORxPID  = (dCORx * P) + (I*Xsum)  + (D*(dCORx-dCORlastX));
    dCORyPID  = (dCORy * P) + (I*Ysum)  + (D*(dCORy-dCORlastY));
    dCORlastX = dCORx;
    dCORlastY = dCORy;
    Xsum      = Xsum+dCORx;
    Ysum      = Ysum+dCORy;

    if ((plane == 0) || (plane == 1)) 
        CMx = CMx - dCORxPID;
    
    if ((plane == 0) || (plane == 2)) 
        CMy = CMy - dCORyPID;
    
    Data_CMx = (CMx % scaleDigitsX) + halfDigits;
    Data_CMy = (CMy % scaleDigitsY) + halfDigits;

    for (char i = 0; i< numCORx; i++) {
        char corPos = DAC_WaveIndexX[i];
        DACout[corPos] = Data_CMx(i);
    }
    for (char i = 0; i< numCORy; i++) {
        char corPos = DAC_WaveIndexY[i];
        DACout[corPos] = Data_CMy(i);
    }
    
    DACout[112] = (loopDir*2500000) + halfDigits;
    DACout[113] = (loopDir*2500000) + halfDigits;
    DACout[114] = (loopDir*2500000) + halfDigits;
    loopDir *= -1;

    t_calc_stop.clock();

    //cout << " " << loopDir <<" "<<status.loopPos<<"\n";

    //cout << "   write DAC" << endl;
    return 0;
}

void sendMessage(const char* Message,const char *error) {
   cout << "Send Messag: " << Message << " Error: " << error << endl;
   return;
   unsigned long pos = CTRL_MEMPOS+100;
   struct t_header { 
        double namesize;
        double sizex;
        double sizey;
        double type;
   } header;
   int thesize = strlen(Message) + strlen(error)+ sizeof(header)*2 + 2;
   unsigned short * mymem = (unsigned short *) malloc(thesize);
   unsigned long structpos = 0;
   mymem[0]=2;  
   header.namesize=7;
   header.sizex = strlen(Message);
   header.sizey = 0;
   header.type = 0; 
   memcpy(mymem+structpos,&header,sizeof(header)); structpos += sizeof(header);
   memcpy(mymem+structpos,"message",7); structpos += 7;
   memcpy(mymem+structpos,Message,strlen(Message)); structpos += strlen(Message);
   header.namesize=5;
   header.sizex = strlen(error);
   header.sizey = 0;
   header.type = 0; 
   memcpy(mymem+structpos,&header,sizeof(header)); structpos += sizeof(header);
   memcpy(mymem+structpos,"error",5); structpos += 5;
   memcpy(mymem+structpos,error,strlen(error)); structpos += strlen(error);
   result = RFM2gWrite( RFM_Handle, pos , &mymem, thesize);
   //free(mymem);
}
    
void Post_error(unsigned char errornr) {
    switch (errornr) {
        case 0: return; break;
        case FOFB_ERROR_ADC   : sendMessage( "FOFB error", "ADC Timeout"); break;
        case FOFB_ERROR_DAC   : sendMessage( "FOFB error", "DAC Problem"); break;
        case FOFB_ERROR_CM100 : sendMessage( "FOFB error", "To much to correct");break;
        case FOFB_ERROR_NoBeam: sendMessage( "FOFB error", "No Current");break;
        case FOFB_ERROR_RMS   : sendMessage( "FOFB error", "Bad RMS");break;
        default               : sendMessage( "FOFB error", "Unknown Problem"); break;
    }
}


void SIGINT_handler(int signum)
{
    cout << endl << "Quit mBox...." << endl;
    init_ADC(0,0);
    ctrl_DAC(0);
    exit(0); 
}

void testDAC() {
    double t_sum=0., t_sum_loop=0., t_sum_all=0.;
    struct timespec t_start, t_stop;

    for (int i = 0; i < (48+64); i++) {
	    DACout[i] = halfDigits;
    }
    ctrl_DAC(1);
    double  loopDir = 1;
    for (int i = 0; i < 15000; i++) {
        if (i%1000==0) {
           cout<<"\n";
           cout<<"time [ms]"<<t_sum<<"\n";
           cout<<"time loop [ms]"<<t_sum_loop<<"\n";
           cout<<"time all [ms]"<<t_sum_all<<endl;
           t_sum_all=t_sum_loop=t_sum=0.;
        }
     	if (readADC()) { cout << " Cant't read ADC Data" << endl; }
        clock_gettime(CLOCK_MODE, &t_start);
    	DACout[112] = (loopDir*2500000) + halfDigits;
	DACout[113] = (loopDir*2500000) + halfDigits;
	DACout[114] = (loopDir*2500000) + halfDigits;
        //cout << DACout[114] << " ";
        loopDir *= -1;
        clock_gettime(CLOCK_MODE, &t_stop);
	result = writeDAC(0 + 0x1C0000);
	if (result) { cout << "Can't write to DAC" << endl; } 	


	usleep(7000);
     }
}


int main() {
    unsigned char errornr;
    cout << "starting MBox " <<  endl << "---------------------" << endl;
    init_rfm();
    if (init_DMA() != 0) 
       exit(1);
    signal(SIGINT, SIGINT_handler);
    cout << "Wait for start" << endl;
    char f_run = 0;     // 0 = IDLE,    1 = RUNNING
    char f_runstate= 0; // 0 = PREINIT; 1 = INITIALIZED; 2 = ERROR

    init_ADC(150,600);
    //read_RFMStruct();
    //testDAC();

    /* timing stats */
    TimingModule * tm=TimingModule::getTimingModule();
    SingleTimer t_start("t_start"), t_stop("t_stop");
    //tm->addDiff("all","t_start","t_stop");
    // all
    tm->addDiff("all   ","t_adc_start","t_dac_stop");
    tm->addDiff("local ","t_adc_start","t_dac_send");
    // ADC
    tm->addDiff("ADC:read   ","t_adc_start","t_adc_read");
    tm->addDiff("ADC:send+cp","t_adc_read","t_adc_stop");
    // CALC
    tm->addDiff("SVD:calc   ","t_calc_start","t_calc_stop");
    // DAC
    tm->addDiff("DAC:enable ","t_dac_start","t_dac_clear");
    tm->addDiff("DAC:write  ","t_dac_clear","t_dac_write");
    tm->addDiff("DAC:send   ","t_dac_write","t_dac_send");
    tm->addDiff("DAC:wait   ","t_dac_send","t_dac_stop");

    int count=1;

    while(1) {
       result = RFM2gRead( RFM_Handle, CTRL_MEMPOS , &f_run , 1 );
       // CHECK IOC 
       if (f_run == 33) {
 		cout << "  !!! MDIZ4T4R was restarted !!! ... Wait for initialization " << endl; 
	        while (f_run != 0) {
       		    result = RFM2gRead( RFM_Handle, CTRL_MEMPOS , &f_run , 1 );
		    sleep(1);
		}
		init_ADC(150,600);
                cout << "Wait for start" << endl;
       }
       // IDLE
       if ((f_run == 0) && (f_runstate == 0)) {}

       // PREPARE CORRECTION
       if ((f_run == 1) && (f_runstate == 0)) {
            read_RFMStruct();
            calcSmat(); 
            f_runstate = 1;
            ctrl_DAC(1);
            cout << "RUN RUN RUN .... " << endl << flush;
       }

       // READ AND CORRECT
       if ((f_run == 1) && (f_runstate == 1)) {
            if (readADC()) { 
              Post_error(FOFB_ERROR_ADC);
            }
            t_start.clock();
            errornr = make_cor();
            if (errornr) {
              Post_error(errornr);
              f_runstate = 2;
            }
            t_stop.clock();
	    if (writeDAC(status.loopPos + 0x1c0000) > 0) {
               Post_error(FOFB_ERROR_DAC);
            }
        tm->eval();
        if (count%1000==0) {
           cout<<"timing summary\n";
           tm->print();
           tm->reset();
           cout<<endl;
         }
         count+=1;
       } 
       // STOP CORRECTION
       if ((f_run == 0) && (f_runstate >= 1)) {
            ctrl_DAC(false);
            cout << "Stopped  ....." << endl << flush;
            f_runstate = 0;
       }
    }
    return 0;
}
