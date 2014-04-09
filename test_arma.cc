#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

int main() {
    double cf = 3.05;

mat A;

A << 1 << 2 << 3 << endr
  << 4 << 5 << 6 << endr;

cout << A << endl;    
    vec ADCDataX;
    vec GainX;
    vec BPMOffsetX;
    ADCDataX   << 2 << 2 << 2 << 2 << 2 << endr;
    GainX      << 2 << 2 << 2 << 2 << 2 << endr ;
    BPMOffsetX << 2 << 2 << 2 << 2 << 2 << endr;

    mat diffX4 = (ADCDataX % GainX * cf * (-1) ) + BPMOffsetX;
    //mat diffX = (ADCDataX .* GainX * cf * (-1)) + BPMOffsetX;
    
    //cout << diffX2 << endl;
    cout << diffX4 << endl;


    return 0;
} 

