#include "map.h"
#include "modules/zmq/logger.h"

Map::Map()
{
}

std::string Map::keyList() {
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
void Map::update(const std::string& key,const std::vector<unsigned char>& value)
{
    m_map[key] = value;
}

void Map::update(const std::string& key, const unsigned char* ptr, const int size)
{
    unsigned char arr[size];
    memcpy(&arr, ptr, size);
    std::vector<unsigned char> v(arr, arr + size);
    update(key, v);
}

void Map::update(const std::string& key, const int value)
{
    int size = sizeof(value);
    update(key, (unsigned char*)&value, size);
}
void Map::update(const std::string& key, const double value)
{
    int size = sizeof(value);
    update(key, (unsigned char*)&value, size);
}
void Map::update(const std::string& key, const std::string& value)
{
    int size = value.size();
    update(key, (unsigned char*)value.data(), size);
}
void Map::update(const std::string& key, const arma::vec& value)
{
    int size = value.n_elem*sizeof(double);
    update(key, (unsigned char*)value.memptr(), size);
}
void Map::update(const std::string& key, const arma::mat& value)
{
    int size = value.n_elem*sizeof(double);
    update(key, (unsigned char*)value.memptr(), size);
}

const std::vector<unsigned char> Map::get(const std::string& key) const
{
    return m_map.at(key);
}

const unsigned char* Map::get_raw(const std::string& key) const
{
    try {
        auto v = m_map.at(key);
        return m_map.at(key).data();
    } catch (std::exception e) {
        Logger::error(_ME_) << e.what() << " [" << key << "] does not exist";
        return nullptr;
    }
}

std::string Map::getAsString(const std::string& key) const
{
    const char* ptr = (const char*) get_raw(key);
    if (ptr == nullptr) {
        return std::string();
    } else {
        return std::string(ptr, get_sizeof(key)/sizeof(char));
    }
}

int Map::getAsInt(const std::string& key) const
{
    int* ptr = (int*) get_raw(key);
    if (ptr == nullptr) {
        return 0;
    } else {
        return *ptr;
    }
}

double Map::getAsDouble(const std::string& key) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return 0;
    } else {
        return *ptr;
    }
}

arma::vec Map::getAsVec(const std::string& key) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return arma::vec();
    } else {
        return arma::vec(ptr, get_sizeof(key)/sizeof(double));
    }
}

arma::mat Map::getAsMat(const std::string& key, int nrows, int ncols) const
{
    double* ptr = (double*) get_raw(key);
    if (ptr == nullptr) {
        return arma::mat();
    } else {
        return arma::mat(ptr, nrows, ncols);
    }
}

const int Map::get_sizeof(const std::string& key) const
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
void Map::selfTest() {
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
