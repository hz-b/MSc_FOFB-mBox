#include <iostream>
#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace arma;

void printthis(const char * c) {
    printf("%s",c);
}


int main() {
    double cf = 3.05;
    printthis ("Hallo Welt");
    
    char stream[] = { 154, 153,153,153,153,153, 3, 64};
    double val;
    memcpy(&val,stream,8);
    cout << val << endl;

    mat SmatX;
    SmatX << 1 << 2 << 3 << 4 << 0 <<  endr << 5 << 6 << 7 << 8 << 0 << endr << 9 << 10 << 11 << 12 << 0 << endr << 1 << 1 << 1 << 1 << 0;
    cout << "SmatX: " << endl << SmatX << endl;
    mat U;
    vec S;
    mat V;
    mat S1;
    mat InvSmatX;

    svd(U,S,V,SmatX);
    cout << "U:" << endl << U << endl << "S:" << endl << S << endl << "V:" << endl << V << endl;

    double ivec = 0.001;
    double mymax = max(S) * ivec;
    char ivecx = 0; //max(find(S > mymax,SmatX.n_rows));
    cout << "max s: " << mymax << endl << "find: " << ivecx << endl;
    S1 = diagmat(S);
    cout << "Diag S: " << endl << S1 << endl;
    cout << "INV S1" << endl << S1.i()<< endl;

    vec dCORx;
    dCORx << 5 << 2 << 99 << 100 << 0;
    double res = max(dCORx);
    cout << "max" << res << endl;
    unsigned int thesize = dCORx.n_elem;
    double astddev = stddev(dCORx);
    res = (thesize-1)* astddev /thesize;
    cout << "rms" << res << endl;

    struct test {
        int a;
        int b;
    };

    return 0;
} 

