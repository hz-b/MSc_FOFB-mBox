/*
    Copyright (C) 2016 Olivier Churlaud <olivier@churlaud.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "extendedmap.h"

#include "modules/zmq/logger.h"

ExtendedMap::ExtendedMap()
{
}

std::string ExtendedMap::keyList() {
    std::string list;
    std::vector<std::string> vect;
    for (const auto& item : m_map) {
        vect.push_back(item.first);
    }
    std::sort (vect.begin(), vect.end());
    for (const std::string& item : vect)
        list.append(item + '\n');

    return list;
}
void ExtendedMap::update(const std::string& key,const std::vector<unsigned char>& value)
{
    m_map[key] = value;
}

void ExtendedMap::update(const std::string& key, const unsigned char* ptr, const int size)
{
    unsigned char arr[size];
    memcpy(&arr, ptr, size);
    std::vector<unsigned char> v(arr, arr + size);
    update(key, v);
}

void ExtendedMap::update(const std::string& key, const int value)
{
    int size = sizeof(value);
    update(key, (unsigned char*)&value, size);
}
void ExtendedMap::update(const std::string& key, const double value)
{
    int size = sizeof(value);
    update(key, (unsigned char*)&value, size);
}
void ExtendedMap::update(const std::string& key, const std::string& value)
{
    int size = value.size();
    update(key, (unsigned char*)value.data(), size);
}
void ExtendedMap::update(const std::string& key, const arma::vec& value)
{
    int size = value.n_elem*sizeof(double);
    update(key, (unsigned char*)value.memptr(), size);
}
void ExtendedMap::update(const std::string& key, const arma::mat& value)
{
    int size = value.n_elem*sizeof(double);
    update(key, (unsigned char*)value.memptr(), size);
}

const std::vector<unsigned char> ExtendedMap::get(const std::string& key) const
{
    return m_map.at(key);
}

const unsigned char* ExtendedMap::get_raw(const std::string& key) const
{
    try {
        auto v = m_map.at(key);
        return m_map.at(key).data();
    } catch (std::exception e) {
        Logger::error(_ME_) << e.what() << " [" << key << "] does not exist";
        return nullptr;
    }
}

std::string ExtendedMap::getAsString(const std::string& key) const
{
    const char* ptr = (const char*) get_raw(key);
    if (ptr == nullptr) {
        return std::string();
    } else {
        return std::string(ptr, get_sizeof(key)/sizeof(char));
    }
}

int ExtendedMap::getAsInt(const std::string& key) const
{
    int* ptr = (int*) get_raw(key);
    if (ptr == nullptr) {
        return 0;
    } else {
        return *ptr;
    }
}

double ExtendedMap::getAsDouble(const std::string& key) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return 0;
    } else {
        return *ptr;
    }
}

arma::vec ExtendedMap::getAsVec(const std::string& key) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return arma::vec();
    } else {
        return arma::vec(ptr, get_sizeof(key)/sizeof(double));
    }
}

arma::mat ExtendedMap::getAsMat(const std::string& key, int nrows, int ncols) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return arma::mat();
    } else {
        return arma::mat(ptr, nrows, ncols);
    }
}

const int ExtendedMap::get_sizeof(const std::string& key) const
{
    int size(0);
    try {
        auto v = m_map.at(key);
        size = v.size();
    } catch(const std::exception& e) {
        Logger::error(_ME_) << e.what();
    }
    return size;
}

#ifndef DEBUG
void ExtendedMap::selfTest() {
    // int
    int i = 5;
    update("i", i);
    int itest = getAsInt("i");
    if (itest == i)
        std::cout << "int\t ok" <<'\n';
    else
        std::cout << "int\t failed" <<'\n';

    // double
    double d =  12.65465;
    update("d", d);
    double dtest = getAsDouble("d");
    if (dtest == d)
        std::cout << "double\t ok" <<'\n';
    else
        std::cout << "double\t failed" <<'\n';

    // string
    std::string s = "this is a string";
    update("s", s);
    std::string stest = getAsString("s");
    if (stest == s)
        std::cout << "string\t ok" <<'\n';
    else
        std::cout << "string\t failed" <<'\n';

    // arma::vec
    arma::vec a = {1, 2, 3.5 };
    update("a", a);
    arma::vec atest = getAsVec("a");
    if (arma::accu(abs(a-atest)) == 0)
        std::cout << "arma::vec ok" <<'\n';
    else
        std::cout << "arma::vec failed" <<'\n';

    // arma::mat
    arma::mat b;
    b << 1 << 3 << 5 << arma::endr
        << 2 << 4 << 6 << arma::endr;
    update("b", b);
    arma::mat btest = getAsMat("b", 2, 3);
    if (arma::accu(abs(b-btest)) == 0)
        std::cout << "arma::mat ok" <<'\n';
    else
        std::cout << "arma::mat failed" <<'\n';

    m_map.erase("i");
    m_map.erase("d");
    m_map.erase("s");
    m_map.erase("a");
    m_map.erase("b");
}
#endif
