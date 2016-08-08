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

#include "rfm_helper.h"

void RFMHelper::dumpMemory(void* data, int len) {
    unsigned char* p = (unsigned char*)data;
    std::printf("'%f'\n", *(double*)p);
    for (int i = 0 ; i < len ; ++i) {
        std::printf("0x%x\n", (unsigned char)*(p+i));
    }
}

void RFMHelper::dumpMemory(volatile void* data, int len) {
    unsigned char* p = (unsigned char*)data;
    std::printf("'%f'\n", *(double*)p);
    for (int i = 0 ; i < len ; ++i) {
        std::printf("0x%x\n", (unsigned char)*(p+i));
    }
}


void RFMHelper::prepareField(std::vector<double>& field, unsigned long pos, unsigned long dim1, unsigned long dim2)
{
    RFM2G_UINT32 threshold = 0;
    RFM2G_UINT32 data_size = dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );
    field = std::vector<double>(dim1*dim2);
    if (data_size<threshold) {
        // use PIO  tranfer
        m_driver->read(pos,(void*)field.data(), data_size);
        //dumpMemory(field.memptr(),8);
    } else {
        // use DMA transfer
        m_driver->read(pos,(void*)m_dma->memory(), data_size);
        // dumpMemory(m_dma->memory(),8);
        field = std::vector<double>((const double *)m_dma->memory(),(const double *)m_dma->memory()+dim1*dim2);
    }
}

void RFMHelper::prepareField(arma::vec& field, unsigned long pos, unsigned long dim1, unsigned long dim2) {
    RFM2G_UINT32 threshold = 0;
    RFM2G_UINT32 data_size = dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    if (data_size<threshold) {
        // use PIO  tranfer
        field.set_size(dim1*dim2);
        m_driver->read(pos,(void*)field.memptr(), data_size);
        //dumpMemory(field.memptr(),8);
    } else {
        // use DMA transfer
        m_driver->read(pos,(void*)m_dma->memory(), data_size);
        // dumpMemory(m_dma->memory(),8);
        field = arma::vec((const double *)m_dma->memory(),dim1*dim2);
    }
}

void RFMHelper::prepareField(arma::mat& field, unsigned long pos, unsigned long dim1, unsigned long dim2)
{
    RFM2G_UINT32 threshold = 0, data_size = dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    if (data_size < threshold) {
        // use PIO transfer
        field.set_size(dim1,dim2);
        m_driver->read(pos, (void*) field.memptr(), data_size);
    } else {
        // use DMA transfer
        m_driver->read(pos, (void*) m_dma->memory(), data_size);
        // dumpMemory(m_dma->memory(),8);
        // dumpMemory(m_dma->memory()+8,8);
        // dumpMemory(m_dma->memory()+16,8);

        field = arma::mat((const double *) m_dma->memory(), dim1, dim2);
    }
    Logger::Logger() << "\t\tm5; " << field(0,0) << " " << field(0,1);
    Logger::Logger() <<"\t\tSize: " << field.n_cols << ":"<< field.n_rows;
}

void RFMHelper::searchField(std::string &name,
                            unsigned long &pos,
                            unsigned long &datasize1,
                            unsigned long &datasize2,
                            unsigned long &datasize
                           )
{
        short header[4];
        m_driver->read(pos, &header, 8); // 4 * 16-Bit = 8
        pos += 8;

        unsigned long namesize = header[0];
        datasize1 = header[1];
        datasize2 = header[2];
        datasize  = datasize1 * datasize2;
        unsigned long type  = header[3];
        if (type == 1)
            datasize *= 8;

        char name_tmp[namesize];
        m_driver->read(pos, name_tmp, namesize);
        name_tmp[namesize] = '\0';
        name = std::string(name_tmp);

        pos += namesize;
}
