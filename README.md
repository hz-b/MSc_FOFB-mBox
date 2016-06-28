# mBox++

## Contents
 * [How to run](#run-howto)
 * [Dependencies](#deps)
 * [How to compile](#compile-howto)
 * [How to generate the documentation] (#docu)
 * [Class organization](#diagram)

## <a name="run-howto"></a> How to run

Read only mode (Corrector values are not sent to the RFM):

    mbox --ro

Full mode (read-write):

    mbox --rw

Experiment-mode:

    mbox --experiment <NAME_OF_PYTHON_FILE>

See `mbox --help` for all the commands.

## <a name="deps"></a> Dependencies
 * Needed libraries
   * cmake
   * build-essential (make, g++, ..)
 * Development package (*-dev)
   * armadillo >= 5 (for Jessie, use `-t jessie-backports` to get version 5)
   * zeroMQ >= 4  (python-zmq is also useful)
   * python (debian: python, libpython, libpython-dev) >= 2.7
   * numpy

For Debian (note: currently used as follow, but consider using the python3
flavours instead):

<pre>
aptitude install build-essential cmake
aptitude install python-zmq libzmq3-dev libpython-dev python-numpy
aptitude install -t jessie-backports armadillo-dev
</pre>

## <a name="compile-howto"></a> How to compile

Go on the root of the project, create a build folder and go in it:

    cd /path/to/mboxpp/
    mkdir build
    cd build

Then configure the compilation. If a dependency is missing, you will be told.

In debug mode with the dummy driver:

    cmake -DCMAKE_BUILD_TYPE=Debug -DDUMMY_DRIVER=ON ..

In debug mode with the normal driver:

    cmake -DCMAKE_BUILD_TYPE=Debug ..

In release mode:

    cmake ..

Then build:

    make

If you want to install (so that it's in the system path):

    make install

To install for the user only (no root needed), add to the cmake command the
following argument (or anything relevant):

    -DCMAKE_INSTALL_PREFIX=$HOME/.local

## <a name="docu"></a> How to generate the documentation

Go to the root of the project and generate the documentation

    cd path/to/mBox++
    doxygen doxygen.conf

The documentation is generated in `doc/html/index.html`

To genereate the LaTeX documentation, go to `doc/latex` and run `make`.

## <a name="diagram"></a> Class organization

![ ](doc/img/mBox_classDiagram.png "Diagramme")
