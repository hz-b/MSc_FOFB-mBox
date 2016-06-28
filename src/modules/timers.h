#ifndef TIMERS_H
#define TIMERS_H

#include <chrono>
#include <map>
#include <cmath>
#include <iostream>

/**
 * @brief Timer that holds various arithmetic values to profile functions.
 *
 * @see See TimingModule for a more comprehensive use
 */
class Timer
{
public:
    enum class Unit : int {
        ns = 0,
        us,
        ms,
        s,
    };  /**< @brief Unit to use */

    /**
     * @brief Constructor
     * @param name Name of the timer (Used for reference in prints).
     */
    explicit Timer(const std::string& name="Timer");

    /**
     * @brief Destructor
     */
    ~Timer();

    /**
     * @brief (Re)Start counting.
     */
    void start();

    /**
     * @brief Calculate duration from last start() and call doArithmetic().
     */
    void stop();

    /**
     * @brief Print the name of the timer, max, min and RMS duration.
     * @param unit The unit in which to display the values
     */
    void print(Unit unit);

    /**
     * @brief Reset all attributes to 0.
     */
    void reset();

    /**
     * @brief Getter for m_timeSpan.
     * @return m_timeSpan value.
     */
    double timeSpan() const;

private:

    /**
     * @brief Do the arithmetic to calculate min, max and diverse sums.
     * @param duration New value to use (in seconds)
     */
    void doArithmetic(double duration);

    /**
     * @brief Calculate the RMS.
     * @return The RMS Value.
     */
    double rms();

    std::string m_name; /**< @brief Timer name */
    std::chrono::steady_clock::time_point m_start; /**< @brief time_point when start() was called */
    double m_timeSpan; /**< @brief Last duration between start() and stop() */
    double m_min; /**< @brief Lesser duration */
    double m_max; /**< @brief Greater duration */
    double m_sum; /**< @brief Sum of each duration */
    double m_sum2; /**< @brief Sum of the square of each duration */
    int m_callNb; /**< @brief How many time stop() was called */
};

/**
 * @brief Map of Timer to manage and print them easily.
 */
class TimerList
{
public:
    /**
     * @brief Constructor
     */
    explicit TimerList(){};
    /**
     * @brief Destructor
     */
    ~TimerList(){};

    /**
     * @brief Add a Timer to the list
     * @param name Name to call the Timer.
     * @param timer Timer to add.
     */
    inline void addTimer(const std::string& name, Timer timer){ m_timerMap[name] = timer;}

    /**
     * @brief Print the timer data. It uses the Timer::print() function plus some
     * formatting.
     *
     * @param unit Unit in which the values should be printed.
     */
    void print(Timer::Unit unit);

    /**
     * @brief Retrieve a timer in the map.
     * @param name Name of the timer to retrieve.
     */
    inline Timer& operator[](const std::string& name) { return m_timerMap[name];}

    /**
     * @brief Count the number of Timer with given name.
     *
     * @param name Name to count,
     * @return Number of Timer with the given name (can only be 0 or 1)
     */
    inline int count(const std::string& name) { return m_timerMap.count(name);}

    /**
     * @brief Empty the map. The TimerList is as newly constructed.
     */
    void reset() { m_timerMap.clear(); }

private:
    std::map<std::string, Timer> m_timerMap; /**< @brief Map containing the Timers */
};

/**
 * @brief Namespace for Timing static functions.
 *
 * To start a timer call
 * ~~~~.cpp
 * TimingModule::addTimer("name_of_timer");
 * ~~~~
 * To stop it:
 * ~~~~.cpp
 * TimingModule::timer("name_of_timer").stop();
 * ~~~~
 * To print its values in ms:
 * ~~~~.cpp
 * TimingModule::timer("name_of_timer").print(Timer::Unit::ms);
 * ~~~~
 * To print all Timer values in ms every 1000 loops:
 * ~~~~.cpp
 * TimingModule::printAll(Timer::Unit::ms, 1000);
 * ~~~~
 */
namespace TimingModule {
    extern TimerList tm;

    /**
     * @brief Global wrapper to TimeList::addTimer() function
     * @param name Name of the new Timer to create or to restart
     */
    inline void addTimer(const std::string& name) {
        if (tm.count(name) > 0) {
            tm[name].start();
            return;
        }
        Timer timer(name);
        tm.addTimer(name, timer);
    }

    /**
     * @brief Access a given Timer.
     * @param name Name of the Timer to access
     */
    inline Timer& timer(const std::string& name) {
        return tm[name];
    }

    /**
     * @brief Print Timers values.
     * @param unit Unit in which the values are printed.
     * @param period Number of cycles before printing.
     */
    inline void printAll(Timer::Unit unit, int period=1) {
        static int i = 0;
        if (i < period) {
            i++;
            return;
        }
        i = 0;
        tm.print(unit);
        tm.reset();
    }
}
#endif // TIMERS_H
