#ifndef RFMDRIVER_H
#define RFMDRIVER_H

#include "rfmdriverinterface.h"

#if DUMMY_RFM_DRIVER
    #include "rfmdriver_dummy.h"
#else

class RFMDriver : public RFMDriverInterface
{
public:
    RFMDriver(RFM2GHANDLE handle) : RFMDriverInterface(handle){};

    /**
     * File Open/Close
     */
    virtual RFM2G_STATUS open(char *devicePath)
    {
        return RFM2gOpen(devicePath, &m_handle);
    };
    virtual RFM2G_STATUS close()
    {
        return RFM2gClose(&m_handle);
    };

    /**
     * Configuration
     */
    virtual RFM2G_STATUS getConfig(RFM2GCONFIG *config)
    {
        return RFM2gGetConfig(m_handle, config);
    };
    virtual RFM2G_STATUS userMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages)
    {
        return RFM2gUserMemory(m_handle, userMemoryPtr, offset, pages);
    };
    virtual RFM2G_STATUS userMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes)
    {
        return RFM2gUserMemoryBytes(m_handle, userMemoryPtr, offset, bytes);
    };
    virtual RFM2G_STATUS unMapUserMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages)
    {
        return RFM2gUserMemory(m_handle, userMemoryPtr, offset, pages);
    };
    virtual RFM2G_STATUS unMapUserMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes)
    {
        return RFM2gUserMemoryBytes(m_handle, userMemoryPtr, offset, bytes);
    };
    virtual RFM2G_STATUS nodeId(RFM2G_NODE *nodeIdPtr)
    {
        return RFM2gNodeID(m_handle, nodeIdPtr);
    };
    virtual RFM2G_STATUS boardId(RFM2G_UINT8 *boardIdPtr)
    {
        return RFM2gBoardID(m_handle, boardIdPtr);
    };
    virtual RFM2G_STATUS size(RFM2G_UINT32 *sizePtr)
    {
        return RFM2gSize(m_handle, sizePtr);
    };
        virtual RFM2G_STATUS first(RFM2G_UINT32 *firstPtr)
    {
        return RFM2gFirst(m_handle, firstPtr);
    };
    virtual RFM2G_STATUS deviceName(char *namePtr)
    {
        return RFM2gDeviceName(m_handle, namePtr);
    };
    virtual RFM2G_STATUS dllVersion(char *versionPtr)
    {
        return RFM2gDllVersion(m_handle, versionPtr);
    };
    virtual RFM2G_STATUS driverVersion(char *versionPtr)
    {
        return RFM2gDriverVersion(m_handle, versionPtr);
    };
    virtual RFM2G_STATUS getDMAThreshold(RFM2G_UINT32 *threshold)
    {
        return RFM2gGetDMAThreshold(m_handle, threshold);
    };
    virtual RFM2G_STATUS setDMAThreshold(RFM2G_UINT32 threshold)
    {
        return RFM2gSetDMAThreshold(m_handle, threshold);
    };
    virtual RFM2G_STATUS setDMAByteSwap(RFM2G_BOOL byteSwap)
    {
        return RFM2gSetDMAByteSwap(m_handle, byteSwap);
    };
    virtual RFM2G_STATUS getDMAByteSwap(RFM2G_BOOL *byteSwap)
    {
        return RFM2gGetDMAByteSwap(m_handle, byteSwap);
    };
    virtual RFM2G_STATUS setPIOByteSwap(RFM2G_BOOL byteSwap)
    {
        return RFM2gSetPIOByteSwap(m_handle, byteSwap);
    };
    virtual RFM2G_STATUS getPIOByteSwap(RFM2G_BOOL *byteSwap)
    {
        return RFM2gGetPIOByteSwap(m_handle, byteSwap);
    };

    /**
     * Data Transferts
     */
    virtual RFM2G_STATUS read(RFM2G_UINT32 offset, void *buffer, RFM2G_UINT32 length)
    {
        return RFM2gRead(m_handle, offset, buffer, length);
    };
    virtual RFM2G_STATUS write(RFM2G_UINT32 offset, void *buffer, RFM2G_UINT32 length)
    {
        return RFM2gWrite(m_handle, offset, buffer, length);
    };
    virtual RFM2G_STATUS peek8(RFM2G_UINT32 offset, RFM2G_UINT8  *value)
    {
        return RFM2gPeek8(m_handle, offset, value);
    };
    virtual RFM2G_STATUS peek16(RFM2G_UINT32 offset, RFM2G_UINT16 *value)
    {
        return RFM2gPeek16(m_handle, offset, value);
    };
    virtual RFM2G_STATUS peek32(RFM2G_UINT32 offset, RFM2G_UINT32 *value)
    {
        return RFM2gPeek32(m_handle, offset, value);
    };
    virtual RFM2G_STATUS poke8(RFM2G_UINT32 offset, RFM2G_UINT8 value)
    {
        return RFM2gPoke8(m_handle, offset, value);	
    };
    virtual RFM2G_STATUS poke16(RFM2G_UINT32 offset, RFM2G_UINT16 value)
    {
        return RFM2gPoke16(m_handle, offset, value);
    };
    virtual RFM2G_STATUS poke32(RFM2G_UINT32 offset, RFM2G_UINT32 value)
    {
        return RFM2gPoke32(m_handle, offset, value);
    };

    // Implemented only on 64 bit Operating Systems 
    virtual RFM2G_STATUS peek64(RFM2G_UINT32 offset, RFM2G_UINT64 *value)
    {
        return RFM2gPeek64(m_handle, offset, value);
    };
    virtual RFM2G_STATUS poke64(RFM2G_UINT32 offset, RFM2G_UINT64 value)
    {
        return RFM2gPoke64(m_handle, offset, value);
    };
    
    /**
     * Interrupt Event Functions
     */
    virtual RFM2G_STATUS enableEvent(RFM2GEVENTTYPE eventType)
    {
        return RFM2gEnableEvent(m_handle, eventType);
    };
    virtual RFM2G_STATUS disableEvent(RFM2GEVENTTYPE eventType)
    {
        return RFM2gDisableEvent(m_handle, eventType);
    };
    virtual RFM2G_STATUS sendEvent(RFM2G_NODE toNode, RFM2GEVENTTYPE eventType, RFM2G_UINT32 extendedData)
    {
        return RFM2gSendEvent(m_handle, toNode, eventType, extendedData);
    };
    virtual RFM2G_STATUS waitForEvent(RFM2GEVENTINFO *eventInfo)
    {
        return RFM2gWaitForEvent(m_handle, eventInfo);
    };
    virtual RFM2G_STATUS enableEventCallback(RFM2GEVENTTYPE eventType, RFM2G_EVENT_FUNCPTR pEventFunc)
    {
        return RFM2gEnableEventCallback(m_handle, eventType, pEventFunc);
    };
    virtual RFM2G_STATUS disableEventCallback(RFM2GEVENTTYPE eventType)
    {
        return RFM2gDisableEventCallback(m_handle, eventType);
    };
    virtual RFM2G_STATUS clearEvent(RFM2GEVENTTYPE eventType)
    {
        return RFM2gClearEvent(m_handle, eventType);
    };
    virtual RFM2G_STATUS cancelWaitForEvent(RFM2GEVENTTYPE eventType)
    {
        return RFM2gCancelWaitForEvent(m_handle, eventType);
    };
    virtual RFM2G_STATUS clearEventCount(RFM2GEVENTTYPE eventType)
    {
        return RFM2gClearEventCount(m_handle, eventType);
    };
    virtual RFM2G_STATUS getEventCount(RFM2GEVENTTYPE eventType, RFM2G_UINT32 *count)
    {
        return RFM2gGetEventCount(m_handle, eventType, count);
    };

    /**
     * Utility
     */
    virtual char* errorMsg(RFM2G_STATUS errorCode)
    {
        return RFM2gErrorMsg(errorCode);
    };
    virtual RFM2G_STATUS getLed(RFM2G_BOOL *led)
    {
        return RFM2gGetLed(m_handle, led);
    };
    virtual RFM2G_STATUS setLed(RFM2G_BOOL led)
    {
        return RFM2gSetLed(m_handle, led);
    };
    virtual RFM2G_STATUS checkRingCont()
    {
        return RFM2gCheckRingCont(m_handle);
    };
    virtual RFM2G_STATUS getDarkOnDark(RFM2G_BOOL *state)
    {
        return RFM2gGetDarkOnDark(m_handle, state);
    };
    virtual RFM2G_STATUS setDarkOnDark(RFM2G_BOOL state)
    {
        return RFM2gSetDarkOnDark(m_handle, state);
    };
    virtual RFM2G_STATUS clearOwnData(RFM2G_BOOL *state)
    {
        return RFM2gClearOwnData(m_handle, state);
    };
    virtual RFM2G_STATUS getTransmit(RFM2G_BOOL *state)
    {
        return RFM2gGetTransmit(m_handle, state);
    };
    virtual RFM2G_STATUS setTransmit(RFM2G_BOOL state)
    {
        return RFM2gSetTransmit(m_handle, state);
    };
    virtual RFM2G_STATUS getLoopback(RFM2G_BOOL *state)
    {
        return RFM2gGetLoopback(m_handle, state);
    };
    virtual RFM2G_STATUS setLoopback(RFM2G_BOOL state)
    {
        return RFM2gSetLoopback(m_handle, state);
    };
    virtual RFM2G_STATUS getParityEnable(RFM2G_BOOL *state)
    {
        return RFM2gGetParityEnable(m_handle, state);
    };
    virtual RFM2G_STATUS setParityEnable(RFM2G_BOOL state)
    {
        return RFM2gSetParityEnable(m_handle, state);
    };
    virtual RFM2G_STATUS getMemoryOffset(RFM2G_MEM_OFFSETTYPE *offset)
    {
        return RFM2gGetMemoryOffset(m_handle, offset);
    };
    virtual RFM2G_STATUS setMemoryOffset(RFM2G_MEM_OFFSETTYPE offset)
    {
        return RFM2gSetMemoryOffset(m_handle, offset);
    };
    virtual RFM2G_STATUS getSlidingWindow(RFM2G_UINT32 *offset, RFM2G_UINT32 *size)
    {
        return RFM2gGetSlidingWindow(m_handle, offset, size);
    };
    virtual RFM2G_STATUS setSlidingWindow(RFM2G_UINT32 offset)
    {
        return RFM2gSetSlidingWindow(m_handle, offset);
    };
};

#endif // DUMMY_RFM_DRIVER

#endif // RFMDRIVER_H
