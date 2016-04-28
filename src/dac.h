#ifndef DAC_H
#define DAC_H

#include <string>
#include <vector>

#include "define.h"

class DMA;
class RFMDriver;

/**
 * @class IOC
 * @brief Represent the Input/Output Controllers the DAC communicate with.
 */
class IOC {
public:

    /**
     * @brief Empty contructor
     */
    explicit IOC() : IOC(0, "", false) {};

    /**
     * @brief Comprehensive contructor
     */
    explicit IOC(int id, std::string name, bool active) { m_id = id; m_name = name; m_active = active; };

    /**
     * @brief Getter for the id
     * @return The id
     */
    int id() { return m_id; };

     /**
      * @brief Getter for the name
      * @return The IOC's name
      */
     std::string name() { return m_name; };

    /**
     * @brief Return whether the IOC is active or not
     * @return True if active
     */
    bool isActive() { return m_active; };

private:

    /**
     * @brief Id of the IOC.
     */
    int m_id;

    /**
     * @brief Name of the IOC.
     */
    std::string m_name;

    /**
     * @brief Whether the IOC is active or not.
     */
    bool m_active;
};


/**
 * @brief This class is used to transmit the data (= corrector values) to the RFM.
 */
class DAC
{
public:
    /**
     * @brief Constructor
     */
    explicit DAC(RFMDriver *driver, DMA *dma);

    /**
     * @brief Enable or disable the DAC and the underlying IOCs.
     *
     * @param status The status can be either:
     *                  * DAC_ENABLE (= 2)
     *                  * DAC_DISABLE (= 1)
     */
    void changeStatus(int status);

    /**
     * @brief Getter for m_waveIndexX element.
     *
     * @param id Id of which we want the position.
     *
     * @return position in the RFM of the requested element.
     */
    double waveIndexXAt(int id) const { return m_waveIndexX.at(id); };

    /**
     * @brief Getter for m_waveIndexY element.
     *
     * @param id Id of which we want the position.
     *
     * @return position in the RFM of the requested element.
     */
    double waveIndexYAt(int id) const { return m_waveIndexY.at(id); };

    /**
     * @brief Setter for m_waveIndexX element.
     *
     * @param vect Vector to copy and save.
     */
    void setWaveIndexX(std::vector<double> vect) { m_waveIndexX = vect; };

    /**
     * @brief Setter for m_waveIndexY element.
     *
     * @param vect Vector to copy and save.
     */
    void setWaveIndexY(std::vector<double> vect) { m_waveIndexY = vect; };

    /**
     * @brief Write to the RFM.
     *
     * @param[in] plane
     * @param[in] loopDir
     * @param[in] data Pointer to the data to write.
     *
     * @return Value of the error (0 = Success)
     */
    int write(double plane, double loopDir, RFM2G_UINT32* data);

private:

    /**
     * @brief Pointer to a DMA object.
     */
    DMA *m_dma;

    /**
     * @brief Pointer to a RFMDriver object.
     */
    RFMDriver *m_driver;

    /**
     * @brief Look-Up Table for indexes: m_waveIndexX[CMx_index] = position of in RFM.
     */
    std::vector<double> m_waveIndexX;

    /**
     * @brief Look-Up Table for indexes: m_waveIndexY[CMy_index] = position of in RFM.
     */
    std::vector<double> m_waveIndexY;

    /**
     * @brief Vector of IOCs to connect and write to.
     */
    std::vector<IOC> m_IOCs;
};

#endif // DAC_H
