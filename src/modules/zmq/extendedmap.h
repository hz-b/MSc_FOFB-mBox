#ifndef EXTENDEDMAP_H
#define EXTENDEDMAP_H

#include <armadillo>

#include <map>

/**
 * @brief Class providing easier access to a std::map container containing various
 * types.
 *
 * All elements are saved as vectors of unsigned char.
 */
class ExtendedMap
{
public:

    /**
     * @brief Constructor
     */
    ExtendedMap();

    /**
     * @brief Return the list of all keys known by the std::map container
     */
    std::string keyList();

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const std::vector<unsigned char>& value);

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const int value);

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const double value);

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const std::string& value);

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const arma::vec& value);

    /**
     * @brief Update the std::map. If the key doesn't exist, it is created.
     *
     * @param key
     * @param value
     */
    void update(const std::string& key, const arma::mat& value);

    /**
     * @brief Update the std::map with a pointer. If the key doesn't exist, it is created.
     *
     * @param key
     * @param ptr
     * @param size Size of the pointer
     */
    void update(const std::string& key, const unsigned char* ptr, const int size);

    /**
     * @brief Get the size of an element.
     * @param key Key of the element which siwe is requested
     *
     * @return Size of the element (0 if the key doesn't exist)
     */
    const int get_sizeof(const std::string& key) const;

    /**
     * @brief Get an element (as it is: vector of unsigned char).
     * @param key Key of the element to return
     *
     * @return std::vector (empty if the key doesn't exist)
     */
    const std::vector<unsigned char> get(const std::string& key) const;

    /**
     * @brief Get an element as a unsigned char* pointer.
     * @param key Key of the element to return
     *
     * @return pointer to the requested element (nullptr if the key doesn't exist)
     */
    const unsigned char* get_raw(const std::string& key) const;

    /**
     * @brief Get an element as an integer.
     * @param key Key of the element to return
     *
     * @return int (0 if the key doesn't exist)
     */
    int getAsInt(const std::string& key) const;

    /**
     * @brief Get an element as a double.
     * @param key Key of the element to return
     *
     * @return double (0 if the key doesn't exist)
     */
    double getAsDouble(const std::string& key) const;

    /**
     * @brief Get an element as a std::string.
     * @param key Key of the element to return
     *
     * @return std::string object (empty if the key doesn't exist)
     */
    std::string getAsString(const std::string& key) const;

    /**
     * @brief Get an element as a arma::vec.
     * @param key Key of the element to return
     *
     * @return arma::vec object (empty if the key doesn't exist)
     */
    arma::vec getAsVec(const std::string& key) const;

    /**
     * @brief Get an element as a arma::mat.
     * @param key Key of the element to return
     * @param nrows Number of rows
     * @param ncols Number of colunms
     *
     * @return arma::mat object (empty if the key doesn't exist)
     */
    arma::mat getAsMat(const std::string& key, int nrows, int ncols) const;

    /**
     * @brief Check if the std::map contains a given key.
     *
     * @param key
     * @return true if it contains the key
     */
    bool has(const std::string& key) { return (m_map.count(key) > 0); };

#ifndef DEBUG
    /**
     * @brief Test all update/getAs functions if DEBUG is true
     */
    void selfTest();
#endif

private:

    /**
     * @brief std::map container we build on.
     */
    std::map<std::string, std::vector<unsigned char> > m_map;
};

#endif // EXTENDEDMAP_H
