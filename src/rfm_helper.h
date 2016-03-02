#ifndef RFM_HELPER_H
#define RFM_HELPER_H

#include <armadillo>
#include <iomanip>
#include <string>
#include <iostream>

#include "dma.h"
#include "rfmdriver.h"
#include "define.h"


class RFMHelper
{
public:
    RFMHelper(RFMDriver *driver, DMA *dma) : m_driver(driver), m_dma(dma){};
    void sendMessage(const char* Message, const char *error);
    void dumpMemory(void* data, int len);
    void dumpMemory(volatile void* data, int len);
    void searchField(std::string &name, unsigned long &pos,
                     unsigned long &datasize1, unsigned long &datasize2, unsigned long &datasize);
    void prepareField(arma::vec& field, unsigned long pos, unsigned long dim1, unsigned long dim2);
    void prepareField(arma::mat& field, unsigned long pos, unsigned long dim1, unsigned long dim2);

    template <class T>
    void prepareField(T &field, unsigned long pos, unsigned long dim1, unsigned long dim2 = 0)
    {
        m_driver->read(pos, (void*) &field, sizeof(field));
    };

    template <class T>
    void readStruct(const std::string structname, T &field, const int tartype)
    {
        unsigned long pos = CTRL_MEMPOS + 1000;
        short elementnr(0);
        std::string name;

        m_driver->read(pos, &elementnr, 2); // 4 * 16-Bit = 8
        pos += 2;

        for(unsigned int i = 0 ; i < elementnr ; i++) {
            unsigned long datasize1(0);
            unsigned long datasize2(0);
            unsigned long datasize(0);
            this->searchField(name, pos, datasize1, datasize2, datasize);

            if (name == structname) {
                std::cout << "\tFound Name: " << name << std::endl;
                this->prepareField(field, pos, datasize1, datasize2);

                return;
            }
            pos += datasize;
        }
        std::cout << "\tWARNING : " << structname << " not found !!!\n";
    };

private:
    RFMDriver *m_driver;
    DMA *m_dma;
};

#endif
