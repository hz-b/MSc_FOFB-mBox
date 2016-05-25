#ifndef RFMDRIVERINTERFACE_H
#define RFMDRIVERINTERFACE_H

#include "config.h"
#if DUMMY_RFM_DRIVER
#include "rfm2g_dummy/rfm2g_api.h"
#else
#include <rfm2g_api.h>
#endif

class RFMDriverInterface
{
public:
    RFMDriverInterface(RFM2GHANDLE handle) : m_handle(handle){};

    /**
     * File Open/Close
     */
    virtual RFM2G_STATUS open(char* devicePath) = 0;
    virtual RFM2G_STATUS close() = 0;

    /**
     * Configuration
     */
    virtual RFM2G_STATUS getConfig(RFM2GCONFIG* config) = 0;
    virtual RFM2G_STATUS userMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages) = 0;
    virtual RFM2G_STATUS userMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes) = 0;
    virtual RFM2G_STATUS unMapUserMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages) = 0;
    virtual RFM2G_STATUS unMapUserMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes) = 0;
    virtual RFM2G_STATUS nodeId(RFM2G_NODE* nodeIdPtr) = 0;
    virtual RFM2G_STATUS boardId(RFM2G_UINT8* boardIdPtr) = 0;
    virtual RFM2G_STATUS size(RFM2G_UINT32* sizePtr) = 0;
    virtual RFM2G_STATUS first(RFM2G_UINT32* firstPtr) = 0;
    virtual RFM2G_STATUS deviceName(char* namePtr) = 0;
    virtual RFM2G_STATUS dllVersion(char* versionPtr) = 0;
    virtual RFM2G_STATUS driverVersion(char* versionPtr) = 0;
    virtual RFM2G_STATUS getDMAThreshold(RFM2G_UINT32* threshold) = 0;
    virtual RFM2G_STATUS setDMAThreshold(RFM2G_UINT32 threshold) = 0;
    virtual RFM2G_STATUS setDMAByteSwap(RFM2G_BOOL byteSwap) = 0;
    virtual RFM2G_STATUS getDMAByteSwap(RFM2G_BOOL* byteSwap) = 0;
    virtual RFM2G_STATUS setPIOByteSwap(RFM2G_BOOL byteSwap) = 0;
    virtual RFM2G_STATUS getPIOByteSwap(RFM2G_BOOL* byteSwap) = 0;

    /**
     * Data Transferts
     */
    virtual RFM2G_STATUS read(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length) = 0;
    virtual RFM2G_STATUS write(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length) = 0;
    virtual RFM2G_STATUS peek8(RFM2G_UINT32 offset, RFM2G_UINT8* value) = 0;
    virtual RFM2G_STATUS peek16(RFM2G_UINT32 offset, RFM2G_UINT16* value) = 0;
    virtual RFM2G_STATUS peek32(RFM2G_UINT32 offset, RFM2G_UINT32* value) = 0;
    virtual RFM2G_STATUS poke8(RFM2G_UINT32 offset, RFM2G_UINT8 value) = 0;
    virtual RFM2G_STATUS poke16(RFM2G_UINT32 offset, RFM2G_UINT16 value) = 0;
    virtual RFM2G_STATUS poke32(RFM2G_UINT32 offset, RFM2G_UINT32 value) = 0;

    // Implemented only on 64 bit Operating Systems
    virtual RFM2G_STATUS peek64(RFM2G_UINT32 offset, RFM2G_UINT64* value) = 0;
    virtual RFM2G_STATUS poke64(RFM2G_UINT32 offset, RFM2G_UINT64 value) = 0;

    /**
     * Interrupt Event Functions
     */
    virtual RFM2G_STATUS enableEvent(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS disableEvent(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS sendEvent(RFM2G_NODE toNode, RFM2GEVENTTYPE eventType, RFM2G_UINT32 extendedData) = 0;
    virtual RFM2G_STATUS waitForEvent(RFM2GEVENTINFO* eventInfo) = 0;
    virtual RFM2G_STATUS enableEventCallback(RFM2GEVENTTYPE eventType, RFM2G_EVENT_FUNCPTR pEventFunc) = 0;
    virtual RFM2G_STATUS disableEventCallback(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS clearEvent(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS cancelWaitForEvent(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS clearEventCount(RFM2GEVENTTYPE eventType) = 0;
    virtual RFM2G_STATUS getEventCount(RFM2GEVENTTYPE eventType, RFM2G_UINT32* count) = 0;

    /**
     * Utility
     */
    virtual char* errorMsg(RFM2G_STATUS errorCode) = 0;
    virtual RFM2G_STATUS getLed(RFM2G_BOOL* led) = 0;
    virtual RFM2G_STATUS setLed(RFM2G_BOOL led) = 0;
    virtual RFM2G_STATUS checkRingCont() = 0;
    virtual RFM2G_STATUS getDarkOnDark(RFM2G_BOOL* state) = 0;
    virtual RFM2G_STATUS setDarkOnDark(RFM2G_BOOL state) = 0;
    virtual RFM2G_STATUS clearOwnData(RFM2G_BOOL* state) = 0;
    virtual RFM2G_STATUS getTransmit(RFM2G_BOOL* state) = 0;
    virtual RFM2G_STATUS setTransmit(RFM2G_BOOL state) = 0;
    virtual RFM2G_STATUS getLoopback(RFM2G_BOOL* state) = 0;
    virtual RFM2G_STATUS setLoopback(RFM2G_BOOL state) = 0;
    virtual RFM2G_STATUS getParityEnable(RFM2G_BOOL* state) = 0;
    virtual RFM2G_STATUS setParityEnable(RFM2G_BOOL state) = 0;
    virtual RFM2G_STATUS getMemoryOffset(RFM2G_MEM_OFFSETTYPE* offset) = 0;
    virtual RFM2G_STATUS setMemoryOffset(RFM2G_MEM_OFFSETTYPE offset) = 0;
    virtual RFM2G_STATUS getSlidingWindow(RFM2G_UINT32* offset, RFM2G_UINT32* size) = 0;
    virtual RFM2G_STATUS setSlidingWindow(RFM2G_UINT32 offset) = 0;

    RFM2GHANDLE handle() const { return m_handle; };

protected:
    RFM2GHANDLE m_handle;
};

#endif // RFMDRIVER_H
