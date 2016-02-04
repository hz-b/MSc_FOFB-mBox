#ifndef DEFINE_H
#define DEFINE_H

#include "config.h"

#define CLOCK_MODE CLOCK_MONOTONIC_RAW

#define CTRL_MEMPOS       0x03000000
#define STATUS_MEMPOS     CTRL_MEMPOS + 50
#define MESSAGE_MEMPOS    CTRL_MEMPOS + 100
#define CONFIG_MEMPOS     CTRL_MEMPOS + 1000

#define readStructtype_pchar    0
#define readStructtype_mat      1
#define readStructtype_vec      2
#define readStructtype_double   3

// ADC
#define ADC_BUFFER_SIZE 256
#define ADC_MEMPOS        0x01000000
#define ADC_TIMEOUT       10000   /* milliseconds */

// DAC
#define DAC_BUFFER_SIZE 128
#define DAC_MEMPOS        0x02000000
#define DAC_TIMEOUT       60000   /* milliseconds */

// DMA
#define DMAOFF_A 0x00100000
  //#define  DMAOFF_A 0xf0000000
  //#define  LINUX_DMA_FLAG 0x0
#define LINUX_DMA_FLAG 0x01
#define LINUX_DMA_FLAG2 0
#define DMA_THRESHOLD 128

#define FOFB_ERROR_ADC      1
#define FOFB_ERROR_DAC      2  
#define FOFB_ERROR_CM100    4 
#define FOFB_ERROR_NoBeam   5 
#define FOFB_ERROR_RMS      6  
#define FOFB_ERROR_ADCReset 8
#define FOFB_ERROR_Unkonwn  7 

#include "rfm2g_api.h"

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
