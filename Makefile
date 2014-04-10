all:
	echo "Not testet, please try make RO first"
	#g++ mBox.cc -o mBox -O2 -larmadillo -Irfm2g/include -Lrfm2g/lib -lrfm2g -lpthread -lrt
RO:
	g++ mBox_RO.cc -g -o mBox_RO -O2 -larmadillo -Irfm2g/include -Lrfm2g/lib -lrfm2g -lpthread -lrt
test-sse: 
	g++ mBox.cc -o mBox -O3 -larmadillo -mfpmath=sse -msse4.2 -march=native -Irfm2g/include -Lrfm2g/lib -lrfm2g -lpthread -lrt
test: test.cc
	g++ test.cc -o test -O2 -larmadillo



