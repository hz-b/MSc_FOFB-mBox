#include "timingmodule.h"

TimingModule * TimingModule::theTimingModule=0;


SingleTimer::SingleTimer(const std::string & _name) {
  name=_name;
  TimingModule * tm = TimingModule::getTimingModule();
  tm->addTimer(this);
}


void TimeDiff::eval() {
  double dt=timing->delta(id_start,id_stop)/ms;
  t_sum+=dt;
  t_sum2+=dt*dt;
  ++n_call;
  if (dt>t_max) t_max=dt;
  if (dt<t_min || t_min==-1.) t_min=dt;
}

std::ostream & operator<<(std::ostream & s, const TimeDiff & td) {
  TimingModule * tm = TimingModule::getTimingModule();
  s<<td.name<<"["<<tm->timerName(td.id_start)<<","<<tm->timerName(td.id_stop)<<"] = "<<td.mean()<<" ("<<td.min()<<","<<td.max()<<")  rms="<<td.rms();
}


void TimingModule::addTimer(SingleTimer* timer) {
  size_t id = all_timer.size();
  all_timer.push_back(timer);
  timer_map[timer->getName()]=id;
  std::cout<<"add Timer "<<timer->getName()<<" "<<id<<std::endl;
}

void TimingModule::addDiff(const std::string &name,
                           const std::string &t_start,
                           const std::string &t_stop)
{
  int a = timer_map[t_start];
  int b = timer_map[t_stop];
  all_diffs.push_back(new TimeDiff(name,a,b,this));
  std::cout<<"add Diff ("<<a<<","<<b<<")"<< std::endl;
}

void TimingModule::eval() {
  for (size_t i=0; i<all_diffs.size();++i) all_diffs[i]->eval();
}

void TimingModule::reset() {
  for (size_t i=0; i<all_diffs.size();++i) all_diffs[i]->reset();
}

void TimingModule::print() {
  for (size_t i=0; i<all_diffs.size();++i) {
    std::cout<<*(all_diffs[i])<<std::endl;
  }
}

double TimingModule::delta(const int a, const int b) {
  return delta(all_timer[a]->getTime(),all_timer[b]->getTime());
}

double TimingModule::delta(const timespec & t_start, const timespec & t_stop) {
  return ( t_stop.tv_sec - t_start.tv_sec )
    + 1.e-9*( t_stop.tv_nsec - t_start.tv_nsec );
}





