#ifndef DEFINE_H
#define DEFINE_H

#include "config.h"
#include "rfm2g_api.h"

#define CLOCK_MODE CLOCK_MONOTONIC_RAW

// See http://wiki.trs.bessy.de/bin/view/OPIhelpdesk/FastOrbitFeedback
const unsigned int CTRL_MEMPOS = 0x03000000;
const unsigned int STATUS_MEMPOS = CTRL_MEMPOS + 50;
const unsigned int MESSAGE_MEMPOS = CTRL_MEMPOS + 100;
const unsigned int CONFIG_MEMPOS = CTRL_MEMPOS + 1000;

const int INJECT_TRIG = 110;
const int TEN_HZ = 62;

// ADC
const int ADC_BUFFER_SIZE = 256;
const int ADC_MEMPOS = 0x01000000;
const int ADC_TIMEOUT = 10000;  /* milliseconds */
const int ADC_STOP = 1;
const int ADC_START = 2;
const int ADC_ENABLE = 3;
const int ADC_NODE = 0x01;

// DAC
const int DAC_BUFFER_SIZE = 128;
const int DAC_MEMPOS = 0x02000000;
const int DAC_TIMEOUT = 60000;   /* milliseconds */
const int DAC_ENABLE = 2;
const int DAC_DISABLE = 1;

// DMA
const int DMAOFF_A = 0x00100000;
  //#define  DMAOFF_A 0xf0000000
  //#define  LINUX_DMA_FLAG 0x0
const int LINUX_DMA_FLAG = 0x01;
const int LINUX_DMA_FLAG2 = 0;
const int DMA_THRESHOLD = 128;

#define FOFB_ERROR_ADC      1
#define FOFB_ERROR_DAC      2
#define FOFB_ERROR_CM100    4
#define FOFB_ERROR_NoBeam   5
#define FOFB_ERROR_RMS      6
#define FOFB_ERROR_ADCReset 8
#define FOFB_ERROR_Unkonwn  7

const int readStructtype_pchar = 0;
const int readStructtype_mat = 1;
const int readStructtype_vec = 2;
const int readStructtype_double = 3;

const RFM2GEVENTTYPE ADC_EVENT = RFM2GEVENT_INTR1;
const RFM2GEVENTTYPE ADC_DAC_EVENT = RFM2GEVENT_INTR2;
const RFM2GEVENTTYPE DAC_EVENT = RFM2GEVENT_INTR3;

extern bool READONLY;

struct t_status {
    unsigned short loopPos;
    unsigned short errornr;
};

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

#endif
