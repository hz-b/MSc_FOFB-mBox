#ifndef DMA_H
#define DMA_H

#include "define.h"

class RFMDriver;

/**
 * @brief Represent the Direct Memory Access.
 */
class DMA
{
public:
    /**
     * @brief Constructor
     */
    explicit DMA();

    /**
     * @brief Destructor
     */
    ~DMA();

    /**
     * @brief Initialize the DMA and register it to the RFM.
     */
    int init(RFMDriver *driver);

    /**
     * @brief Direct access to the `m_memory` pointer.
     * @return `m_memory` pointer
     */
    volatile char* memory() { return m_memory; };

    /**
     * @brief Direct access to `m_status`.
     * @return m_status
     */
    t_status* status() { return m_status; };

private:
    /**
     * @brief Pointer to DMA memory.
     *
     * @note DMA Buffer pointer = pDmaCard
     */
    volatile char *m_memory;

    /**
     * @brief Status
     */
    t_status *m_status;
};

#endif // DMA_H
