#ifndef TIMERS_H
#define TIMERS_H

#include <chrono>
#include <map>
#include <cmath>
#include <iostream>
#include <ratio>
class Timer
{
public:
    enum class Unit : int {
        ns = 0,
        us,
        ms,
        s,
    };
    explicit Timer(const std::string& name="Timer");
    ~Timer();
    void start();
    void stop();
    void print(Unit unit);
    double rms();
    void reset();
    double timeSpan() const;

private:
    void doArithmetic(double duration);
    std::string m_name;
    std::chrono::steady_clock::time_point m_start;
    double m_timeSpan;
    double m_min;
    double m_max;
    double m_sum;
    double m_sum2;
    int m_callNb;

};
class TimerList
{
public:
    explicit TimerList(){};
    ~TimerList(){};
    inline void addTimer(const std::string& name, Timer timer){ m_timerMap[name] = timer;}
    void print(Timer::Unit unit);
    inline Timer& operator[](const std::string& name) { return m_timerMap[name];}
    inline int count(const std::string& name) { return m_timerMap.count(name);}
    void reset() { m_timerMap.clear(); }

private:
    std::map<std::string, Timer> m_timerMap;
};

namespace TimingModule {
    extern TimerList tm;
    inline void addTimer(const std::string& name) {
        if (tm.count(name) > 0) {
            tm[name].start();
            return;
        }
        Timer timer(name);
        tm.addTimer(name, timer);
    }
    inline Timer& timer(const std::string& name) {
        return tm[name];
    }
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
