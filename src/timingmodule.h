#ifndef TIMINGMODULE_HP
#define TIMINGMODULE_H

#include <iostream>
#include <ctime>
#include <map>
#include <armadillo>

#define CLOCK_MODE CLOCK_MONOTONIC_RAW

class TimingModule;

class SingleTimer {
public:
  std::string name;
  struct timespec t_start;
public:
  SingleTimer(const std::string &_name);
  void clock() {
    clock_gettime(CLOCK_MODE, &t_start);
  }
  static void wait(long ns) {
    struct timespec t_stop;
    /*
      t_stop.tv_sec=t_start.tv_sec;
      t_stop.tv_nsec=t_start.tv_nsec+ns;
      const long onesec=1000000000;
      if (t_stop.tv_nsec>=onesec) {
      t_stop.tv_sec+=1;
      t_stop.tv_nsec-=onesec;
      }
    */
    t_stop.tv_sec=0;
    t_stop.tv_nsec=ns;
    clock_nanosleep(CLOCK_MODE, 0, &t_stop, 0);
  }
  const timespec getTime() { return t_start; }
  const std::string & getName() { return name; }
};

template <class T>
T sqr(const T a) { return a*a; }

class TimeDiff {
  std::string name;
  double t_sum, t_sum2;
  double t_min, t_max;
  long   n_call;
  int    id_start,id_stop;
  TimingModule * timing;
  static const double ms=1.e-3;
public:
  TimeDiff(const std::string & _name, int a, int b, TimingModule * tm) {
    name=_name;
    id_start=a;
    id_stop=b;
    timing=tm;
    reset();
  }
  void reset() {
    t_sum=t_sum2=0.;
    n_call=0;
    t_min=-1.;
    t_max=-1.;
  }
  void eval();
  double mean() const {
    return t_sum/n_call;
  }
  double mean2() const {
    return t_sum2/n_call;
  }
  double min() const {
    return t_min;
  }
  double max() const {
    return t_max;
  }
  double rms() const {
    return sqrt( mean2() - sqr(mean()) );
  }
  friend std::ostream & operator<<(std::ostream & s, const TimeDiff & td);
};

class TimingModule {
  static TimingModule *theTimingModule;
  std::map<std::string, int>     timer_map;
  std::vector<SingleTimer*> all_timer;
  std::vector<TimeDiff*>    all_diffs;

public:
  static TimingModule * getTimingModule() { if (!theTimingModule) theTimingModule=new TimingModule(); return theTimingModule; }
  void addTimer(SingleTimer *);
  void addDiff(const std::string &name, const std::string &t_start, const std::string  &t_stop);
  void eval();
  void print();
  void reset();
  const std::string & timerName(const int id) { return all_timer[id]->getName(); }
  double delta(const int id_start, const int id_stop);
  static double delta(const timespec & t_start, const timespec & t_stop);
};

#endif // TIMINGMODULE_H
