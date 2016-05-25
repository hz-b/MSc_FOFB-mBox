#ifndef RFMDRIVER_DUMMY_H
#define RFMDRIVER_DUMMY_H

#include "rfmdriverinterface.h"

class RFMDriver : public RFMDriverInterface
{
public:
    RFMDriver(RFM2GHANDLE handle) : RFMDriverInterface(handle){};

    /**
     * File Open/Close
     */
    virtual RFM2G_STATUS open(char* devicePath);
    virtual RFM2G_STATUS close(){ return RFM2G_SUCCESS; };

    /**
     * Configuration
     */
    virtual RFM2G_STATUS getConfig(RFM2GCONFIG* config){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS userMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS userMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS unMapUserMemory(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 pages){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS unMapUserMemoryBytes(volatile void** userMemoryPtr, RFM2G_UINT64 offset, RFM2G_UINT32 bytes){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS nodeId(RFM2G_NODE* nodeIdPtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS boardId(RFM2G_UINT8* boardIdPtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS size(RFM2G_UINT32* sizePtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS first(RFM2G_UINT32* firstPtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS deviceName(char* namePtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS dllVersion(char* versionPtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS driverVersion(char* versionPtr){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getDMAThreshold(RFM2G_UINT32* threshold);
    virtual RFM2G_STATUS setDMAThreshold(RFM2G_UINT32 threshold);
    virtual RFM2G_STATUS setDMAByteSwap(RFM2G_BOOL byteSwap){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getDMAByteSwap(RFM2G_BOOL* byteSwap){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setPIOByteSwap(RFM2G_BOOL byteSwap){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getPIOByteSwap(RFM2G_BOOL* byteSwap){ return RFM2G_SUCCESS; };

    /**
     * Data Transferts
     */
    virtual RFM2G_STATUS read(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length);
    virtual RFM2G_STATUS write(RFM2G_UINT32 offset, void* buffer, RFM2G_UINT32 length);

    virtual RFM2G_STATUS peek8(RFM2G_UINT32 offset, RFM2G_UINT8* value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS peek16(RFM2G_UINT32 offset, RFM2G_UINT16* value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS peek32(RFM2G_UINT32 offset, RFM2G_UINT32* value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS poke8(RFM2G_UINT32 offset, RFM2G_UINT8 value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS poke16(RFM2G_UINT32 offset, RFM2G_UINT16 value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS poke32(RFM2G_UINT32 offset, RFM2G_UINT32 value){ return RFM2G_SUCCESS; };

    // Implemented only on 64 bit Operating Systems
    virtual RFM2G_STATUS peek64(RFM2G_UINT32 offset, RFM2G_UINT64* value){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS poke64(RFM2G_UINT32 offset, RFM2G_UINT64 value){ return RFM2G_SUCCESS; };

    /**
     * Interrupt Event Functions
     */
    virtual RFM2G_STATUS enableEvent(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS disableEvent(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS sendEvent(RFM2G_NODE toNode, RFM2GEVENTTYPE eventType, RFM2G_UINT32 extendedData){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS waitForEvent(RFM2GEVENTINFO* eventInfo);
    virtual RFM2G_STATUS enableEventCallback(RFM2GEVENTTYPE eventType, RFM2G_EVENT_FUNCPTR pEventFunc){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS disableEventCallback(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS clearEvent(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS cancelWaitForEvent(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS clearEventCount(RFM2GEVENTTYPE eventType){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getEventCount(RFM2GEVENTTYPE eventType, RFM2G_UINT32* count){ return RFM2G_SUCCESS; };

    /**
     * Utility
     */
    virtual char* errorMsg(RFM2G_STATUS errorCode){ const char* error = "this is a dummy driver"; return const_cast<char*>(error); };
    virtual RFM2G_STATUS getLed(RFM2G_BOOL* led){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setLed(RFM2G_BOOL led){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS checkRingCont(){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getDarkOnDark(RFM2G_BOOL* state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setDarkOnDark(RFM2G_BOOL state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS clearOwnData(RFM2G_BOOL* state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getTransmit(RFM2G_BOOL* state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setTransmit(RFM2G_BOOL state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getLoopback(RFM2G_BOOL* state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setLoopback(RFM2G_BOOL state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getParityEnable(RFM2G_BOOL* state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setParityEnable(RFM2G_BOOL state){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getMemoryOffset(RFM2G_MEM_OFFSETTYPE* offset){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setMemoryOffset(RFM2G_MEM_OFFSETTYPE offset){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS getSlidingWindow(RFM2G_UINT32* offset, RFM2G_UINT32* size){ return RFM2G_SUCCESS; };
    virtual RFM2G_STATUS setSlidingWindow(RFM2G_UINT32 offset){ return RFM2G_SUCCESS; };

private:
    RFM2G_UINT32 m_DMAthreshold;
};

#endif // RFMDRIVER_H
