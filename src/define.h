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

#ifndef DEFINE_H
#define DEFINE_H

#include "error.h"
#include "config.h"
#if DUMMY_RFM_DRIVER
#include "rfm2g_dummy/rfm2g_api.h"
#else
#include "rfm2g_api.h"
#endif

const char DEVICE_NAME[] = "/dev/rfm2g0";               /**< @brief RFM device name. */
const bool WEIGHTED_CORR = true;                        /**< @brief Do we use a weighted correction? */

// See http://wiki.trs.bessy.de/bin/view/OPIhelpdesk/FastOrbitFeedback
const unsigned int CTRL_MEMPOS = 0x03000000;            /**< @brief Memory position of the Control register. */
const unsigned int STATUS_MEMPOS = CTRL_MEMPOS + 50;    /**< @brief Memory position of the Status register. */
const unsigned int MESSAGE_MEMPOS = CTRL_MEMPOS + 100;  /**< @brief Memory position of the Message register. */
const unsigned int CONFIG_MEMPOS = CTRL_MEMPOS + 1000;  /**< @brief Memory position of the Config register. */

const int INJECT_TRIG = 110;                            /**< @brief Index of the injection BPM in ADC. */
const int TEN_HZ = 62;                                  /**< @brief Index of the 10Hz BPM in ADC. */
const int HBP1D5R = 142;                                /**< @brief Index of the HBP1D5R BPM in ADC. */

// ADC
const int ADC_BUFFER_SIZE = 256;                        /**< @brief ADC Buffer size. */
const int ADC_MEMPOS = 0x01000000;                      /**< @brief Memory position of the ADC register. */
const int ADC_TIMEOUT = 10000;                          /**< @brief Timeout duration when waiting for ADC interruption (in ms). */
const int ADC_STOP = 1;                                 /**< @brief Command to stop the ADC. */
const int ADC_START = 2;                                /**< @brief Command to start the ADC. */
const int ADC_ENABLE = 3;                               /**< @brief Command to enable the ADC. */
const int ADC_NODE = 0x01;                              /**< @brief ADC Node. */

// DAC
const int DAC_BUFFER_SIZE = 128;                        /**< @brief DAC Buffer size. */
const int DAC_MEMPOS = 0x02000000;                      /**< @brief Memory position of the DAC register. */
const int DAC_TIMEOUT = 60000;                          /**< @brief Timeout duration when waiting for DAC interruption (in ms). */
const int DAC_ENABLE = 2;                               /**< @brief Command to enable the DAC. */
const int DAC_DISABLE = 1;                              /**< @brief Command to disable the DAC. */

// DMA
const int DMAOFF_A = 0x00100000;                        /**< @brief DMA ?????. */
  //#define  DMAOFF_A 0xf0000000
  //#define  LINUX_DMA_FLAG 0x0
const int LINUX_DMA_FLAG = 0x01;                        /**< @brief DMA flag ??. */
const int LINUX_DMA_FLAG2 = 0;                          /**< @brief DMA flag2 ??. */
const int DMA_THRESHOLD = 128;                          /**< @brief Threshold after which DMA is used. */

const RFM2GEVENTTYPE ADC_EVENT = RFM2GEVENT_INTR1;      /**< @brief Interruption for ADC. */
const RFM2GEVENTTYPE ADC_DAC_EVENT = RFM2GEVENT_INTR2;  /**< @brief Interruption for ADC and DAC. */
const RFM2GEVENTTYPE DAC_EVENT = RFM2GEVENT_INTR3;      /**< @brief Interruption for DAC. */

extern bool READONLY;                                   /**< @brief Are we readonly? */

/**
 * @brief Status structure
 */
struct t_status {
    unsigned short loopPos; /**< @brief Loop number (1-512) */
    unsigned short errornr; /**< @brief Error, see Error::ErrorCode. */
};

#endif
