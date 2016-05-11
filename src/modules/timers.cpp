#include "modules/timers.h"

#include <iomanip>

Timer::Timer(const std::string& name)
    : m_min((double)0)
    , m_max(0)
    , m_callNb(0)
    , m_name(name)
{
    m_start = std::chrono::steady_clock::now();
}

Timer::~Timer()
{}

void Timer::start()
{
    m_start = std::chrono::steady_clock::now();
}

void Timer::stop()
{
    using namespace std::chrono;

    steady_clock::time_point stop = steady_clock::now();
    duration<double> duration_time = duration_cast<duration<double> >(stop - m_start);
    m_timeSpan = duration_time.count();
    this->doArithmetic(m_timeSpan);
}

void Timer::doArithmetic(double duration)
{
    if (m_min > duration || m_min == 0) {
        m_min = duration;
    }
    if (m_max < duration || m_max == 0) {
        m_max = duration;
    }
    m_sum += duration;
    m_sum2 += duration*duration;
    m_callNb++;
}

double Timer::rms()
{
    return std::sqrt( m_sum2/m_callNb - std::pow( m_sum/m_callNb, 2));
}

double Timer::timeSpan() const
{
    return m_timeSpan;
}

void Timer::reset()
{
    m_min = m_max = 0;
    m_callNb = 0;
    m_sum = m_sum2 = 0;
}

void Timer::print(Unit unit)
{
    double coef = 1;
    std::string unitName(" ");
    switch (unit)
    {
        case Unit::ns:
            coef = 1e9;
            unitName += "ns";
            break;
        case Unit::us:
            coef = 1e6;
            unitName += "us";
            break;
        case Unit::ms:
            coef = 1e3;
            unitName += "ms";
            break;
        default:
            coef = 1;
            unitName += "s";
            break;
    }
    std::cout << '['<< m_name << ']' << '\t'
              << "Min: " << m_min*coef << unitName << " -- "
              << "Max: " << m_max*coef << unitName << " -- "
              << "RMS: " << rms()*coef << unitName <<'\n';
}

void TimerList::print(Timer::Unit unit)
 {
    std::cout << "==================" <<'\n';
    for (auto t : m_timerMap) {
        std::cout << " ";
        t.second.print(unit);
    }
    std::cout << "==================" <<'\n';
};
