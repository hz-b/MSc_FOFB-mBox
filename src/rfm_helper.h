#ifndef RFM_HELPER_H
#define RFM_HELPER_H

#include <armadillo>
#include <iomanip>
#include <string>

#include "dma.h"
#include "rfmdriver.h"
#include "define.h"
#include "modules/zmq/logger.h"

class RFMHelper
{
public:
    /**
     * @brief Constructor.
     *
     * @param driver Pointer to a RFMDriver object
     * @param dma Pointer to a DMA object
     */
    RFMHelper(RFMDriver *driver, DMA *dma) : m_driver(driver), m_dma(dma){};

    /**
     * @brief Dump a pointer data
     *
     * @param data Pointer to the data to dump
     * @param len Length of the data
     */
    void dumpMemory(void* data, int len);

    /**
     * @brief Dump a pointer data
     *
     * @param data Pointer to the data to dump
     * @param len Length of the data
     */
    void dumpMemory(volatile void* data, int len);

    /**
     * @brief Search a structure in the RFM.
     *
     * @param[in] name name of the structure
     * @param[out] position Position of the RFM where the field begins
     * @param[out] dim1 1st dimension of the variable
     * @param[out] dim2 2nd dimension of the variable
     */
    void searchField(std::string &name, unsigned long &pos,
                     unsigned long &datasize1, unsigned long &datasize2, unsigned long &datasize);

    /**
     * @brief Fill a variable from a given position in the RFM.
     *
     * @param[out] field Variable to fill
     * @param[in] position Position of the RFM where the value begins
     * @param[in] dim1 1st dimension of the variable
     * @param[in] dim2 2nd dimension of the variable
     */
    void prepareField(arma::vec& field, unsigned long pos, unsigned long dim1, unsigned long dim2);

    /**
     * @brief Fill a variable from a given position in the RFM.
     *
     * @param[out] field Variable to fill
     * @param[in] position Position of the RFM where the value begins
     * @param[in] dim1 1st dimension of the variable
     * @param[in] dim2 2nd dimension of the variable
     */
    void prepareField(arma::mat& field, unsigned long pos, unsigned long dim1, unsigned long dim2);

    /**
     * @brief Fill a variable from a given position in the RFM.
     *
     * @param[out] field Variable to fill
     * @param[in] position Position of the RFM where the value begins
     * @param[in] dim1 1st dimension of the variable
     * @param[in] dim2 2nd dimension of the variable
     */
    void prepareField(std::vector<double>& field, unsigned long pos, unsigned long dim1, unsigned long dim2);

    /**
     * @brief Fill a variable from a given position in the RFM.
     *
     * @param[out] field Variable to fill
     * @param[in] position Position of the RFM where the value begins
     * @param[in] dim1 (not used)
     * @param[in] dim2 (not used)
     */
    template <class T>
    void prepareField(T &field, unsigned long pos, unsigned long dim1, unsigned long dim2 = 0)
    {
        m_driver->read(pos, (void*) &field, sizeof(field));
    };

    /**
     * @brief Read a structure from the RFM.
     *
     * @param[in] structname Name of the structure to read.
     * @param[out] field Variable to fill
     * @param[in] tartype Type of variable
     */
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
                Logger::Logger() << "\tFound Name: " << name;
                this->prepareField(field, pos, datasize1, datasize2);

                return;
            }
            pos += datasize;
        }
        Logger::Logger() << "\tWARNING : " << structname << " not found !!!";
    };

private:

    /**
     * @brief Poitner to a RFMDriver object
     */
    RFMDriver *m_driver;

    /**
     * @brief Pointer to a DMA object.
     */
    DMA *m_dma;
};

#endif
