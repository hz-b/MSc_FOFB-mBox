all:
	g++ mBox.cc -o mBox -O2 -larmadillo -Irfm2g/include -Lrfm2g/lib -lrfm2g -lpthread -lrt
test-sse: 
	g++ mBox.cc -o mBox -O3 -larmadillo -mfpmath=sse -msse4.2 -march=native -Irfm2g/include -Lrfm2g/lib -lrfm2g -lpthread -lrt
test: test.cc
	g++ test.cc -o test -O2 -larmadillo
package: 
	mkdir package
	cp mBox package
	cp /usr/lib/libblas.so.3gf package
	cp /usr/lib/liblapack.so.3gf package
	cp /usr/lib/libgfortran.so.3 package
	cp /lib/i686/cmov/libc.so.6 package
	cp arma/usr/lib/libarmadillo.so package/libarmadillo.so.0
	tar -czf mBox_package.tar.gz package
	rm -rf package



