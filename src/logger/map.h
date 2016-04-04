#ifndef MAP_H
#define MAP_H

#include <armadillo>

#include <map>

class Map
{
public:
    Map();
    std::string keyList();
    void update(const std::string& key, const std::vector<unsigned char>& value);
    void update(const std::string& key, const int value);
    void update(const std::string& key, const double value);
    void update(const std::string& key, const std::string& value);
    void update(const std::string& key, const arma::vec& value);
    void update(const std::string& key, const arma::mat& value);
    void update(const std::string& key, const unsigned char* ptr, const int size);

    const int get_sizeof(const std::string& key) const;

    const std::vector<unsigned char> get(const std::string& key) const;
    const unsigned char* get_raw(const std::string& key) const;
    int getAsInt(const std::string& key) const;
    double getAsDouble(const std::string& key) const;
    std::string getAsString(const std::string& key) const;
    arma::vec getAsVec(const std::string& key) const;
    arma::mat getAsMat(const std::string& key, int nrows, int ncols) const;

    bool has(const std::string& key) { return (m_map.count(key) > 0); };
#ifndef DEBUG
    void selfTest();
#endif

private:
    std::map<std::string, std::vector<unsigned char> > m_map;
};

#endif // MAP_H
