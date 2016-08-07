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
