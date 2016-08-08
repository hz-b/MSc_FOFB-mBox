/*
    Copyright (C) 2015 Andreas Schälicke <andreas.schaelicke@helmholtz-berlin.de>
    Copyright (C) 2015 Dennis Engel <dennis.brian.engel@googlemail.com>
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

#include "adc.h"

#include <chrono>
#include <thread>

#include "dma.h"
#include "modules/zmq/logger.h"
#include "rfmdriver.h"


ADC::ADC(RFMDriver *driver, DMA *dma)
    : m_driver(driver)
    , m_dma(dma)
    , m_node(ADC_NODE)
    , m_buffer(ADC_BUFFER_SIZE)
{}

ADC::~ADC()
{

}

int ADC::init()
{
    if (READONLY)
        return 0;

    Logger::Logger() << "Init ADC";

    RFM2G_INT32 ctrlBuffer[128];
//    short Navr = DAC_freq/freq;
    ctrlBuffer[0] = 512; // RFM2G_LOOP_MAX
    ctrlBuffer[1] = 0;
    ctrlBuffer[2] = 0;
    ctrlBuffer[3] = 2;   // Navr


    // Write ADC CTRL
    Logger::Logger() << "\tADC write sampling config";

    RFM2G_UINT32 threshold = 0;
    // see if DMA threshold and buffer are intialized
    m_driver->getDMAThreshold(&threshold);

    int data_size = 512;
    RFM2G_STATUS writeError;
    if (data_size < threshold) {
       // use PIO transfer
       writeError = m_driver->write(0, &ctrlBuffer, 512);
    } else {
       RFM2G_INT32 *dst = (RFM2G_INT32*)m_dma->memory();
       for (int i = 0 ; i < 128 ; i++) {
           dst[i] = ctrlBuffer[i];
       }
       writeError = m_driver->write(0, (void*)m_dma->memory(), data_size);
    }
    if (writeError) {
        Logger::error(_ME_) << "Can't write ADC config: " << m_driver->errorMsg(writeError);
        return 1;
    }

    // Enable ADC
    Logger::Logger() << "\tADC enable sampling";
    RFM2G_STATUS sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_ENABLE);
    if (sendEventError) {
        Logger::error(_ME_) << "Can't enable ADC: " << m_driver->errorMsg(sendEventError);
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    // Start Sampling
    Logger::Logger() << "\tADC start sampling.";
    sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_START);
    if (sendEventError) {
        Logger::error(_ME_) << "Can't start sampling: " << m_driver->errorMsg(sendEventError);
        return 1;
    }
    Logger::Logger() << "\tADC started";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
}

int ADC::stop()
{
    if (READONLY)
        return 0;

    Logger::Logger() << "ADC stoping sampling....";

    RFM2G_STATUS sendEventError = m_driver->sendEvent(m_node, ADC_DAC_EVENT, ADC_STOP);
    if (sendEventError) {
        Logger::error(_ME_) << "Can't stop ADC: " << m_driver->errorMsg(sendEventError);
        return 1;
    }
    Logger::Logger() << "\tADC stopped.";

    return 0;
}


int ADC::read()
{
    RFM2GEVENTINFO eventInfo;              // Info about received interrupts
    eventInfo.Event = ADC_EVENT;           // We'll wait on this interrupt
    eventInfo.Timeout = ADC_TIMEOUT;       // We'll wait this many milliseconds

    // Wait on an interrupt from the other Reflective Memory board
    RFM2G_STATUS waitError = this->waitForEvent(eventInfo);
    if (waitError) {
        Logger::error(_ME_) << "waitForEvent:" << m_driver->errorMsg(waitError);
        return 1;
    }

    if ( m_dma->status() == NULL ) {
        Logger::error(_ME_) << "Null status";
        return 1;
    }

    m_dma->status()->loopPos = eventInfo.ExtendedInfo;
    RFM2G_NODE otherNodeId = eventInfo.NodeId;

    /* Now read data from the other board from BPM_MEMPOS */
    RFM2G_UINT32 threshold = 0;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    int data_size = ADC_BUFFER_SIZE  *sizeof( RFM2G_INT16 );

    if (data_size  < threshold) {
        // use PIO transfer
        RFM2G_STATUS readError = m_driver->read(ADC_MEMPOS + ( m_dma->status()->loopPos * data_size),
                                                (void*)m_buffer.data(), data_size);
        if (readError) {
            Logger::error(_ME_) << "Read error: " << m_driver->errorMsg(readError);

            return 1;
        }

    } else {
        RFM2G_STATUS readError = m_driver->read(ADC_MEMPOS + ( m_dma->status()->loopPos * data_size),
                                                (void*) m_dma->memory(), data_size);
        if (readError) {
            Logger::error(_ME_) << "Read error DMA: " << m_driver->errorMsg(readError);

            return 1;
        }

        RFM2G_INT16 *src = (RFM2G_INT16*) m_dma->memory();
        for (int i = 0 ; i < ADC_BUFFER_SIZE ; ++i) {
            m_buffer.at(i) = src[i];
        }
    }

    // Send an interrupt to the IOC Reflective Memory board
    if (!READONLY) {
        RFM2G_STATUS sendEventError = m_driver->sendEvent(otherNodeId, ADC_EVENT, 0);
        if (sendEventError) {
            Logger::error(_ME_) << "sendEvent: " << m_driver->errorMsg(sendEventError);
            return 1;
        }
    }
    return 0;
}

RFM2G_STATUS ADC::waitForEvent(RFM2GEVENTINFO &eventInfo)
{
    RFM2G_STATUS eventError = m_driver->clearEvent(eventInfo.Event);
    if (eventError) {
        Logger::error(_ME_) << "in clearError, error: " << eventError << '\n';
        return eventError;
    }
    eventError = m_driver->enableEvent(eventInfo.Event);
    if (eventError) {
        Logger::error(_ME_) << "in enableEvent, error: " << eventError << '\n';
        return eventError;
    }

    RFM2G_STATUS waitForEventError = m_driver->waitForEvent(&eventInfo);
    return waitForEventError;
}
