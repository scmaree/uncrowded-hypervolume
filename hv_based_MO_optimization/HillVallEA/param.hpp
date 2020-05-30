#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "hillvallea_internal.hpp"


// start the eda namespace
namespace hillvallea
{
  
  // Column vectors
  //----------------------------------------------------------------------------
  class vec_t : public std::vector<double> {
    
  public:
    
    // constructors
    vec_t();
    vec_t(const size_t & n);
    vec_t(const size_t & n, const double & val);
    
    // return as array
    double * toArray();
    void setRaw(double * raw, const size_t & size);

    
    // initializations
    void fill(const double & d);
    vec_t & operator=(const vec_t& vr);
    vec_t & operator+=(const vec_t& vr);
    vec_t & operator-=(const vec_t& vr);
    vec_t & operator+=(const double& val);
    vec_t & operator-=(const double& val);
    vec_t & operator*=(const double& val);
    vec_t & operator/=(const double& val);
    
    // math functions
    double dot(vec_t & v2);       // not const because we use toArray().
    double norm() const;          // Euclid norm, equal to p.norm() = sqrt(p.squaredNorm());
    double squaredNorm() const;   //
    double infinitynorm() const;  //
    double variance() const;      // variance of the vector. 
    double mean() const;          // mean of the vector
    double sum() const;           // sum of the vector;
    double prod() const;          // product of the vector entries;
    double max_elem() const;
    double min_elem() const; 
    double percentile(const double percent) const;
    vec_t abs() const; // returns copy of vector with absolute values;
    
  };
  
  // overloaded operators for vectors
  vec_t operator-(const vec_t &pl, const vec_t &pr);
  vec_t operator-(const double &val, const vec_t &p);
  vec_t operator-(const vec_t &p, const double &val);
  vec_t operator+(const vec_t &pl, const vec_t &pr);
  vec_t operator+(const double &val, const vec_t &p);
  vec_t operator+(const vec_t &p, const double &val);
  vec_t operator*(const double &val, const vec_t &p);
  vec_t operator*(const vec_t &p, const double &val);
  vec_t operator/(const vec_t &p, const double &val);
  std::ostream &operator<<(std::ostream &os, vec_t const &p);
  
  
  
  // matrices
  //----------------------------------------------------------------------------
  class matrix_t {
    
  public:
    
    // constructors & destructors
    matrix_t();
    matrix_t(const size_t &n, const size_t &m);
    matrix_t(const size_t &n, const size_t &m, const double &val);
    matrix_t(const matrix_t &m); // copy constructor
    ~matrix_t();

    
    // assignment operator
    matrix_t & operator=(const matrix_t &m);
    matrix_t & operator+=(const matrix_t& vr);
    matrix_t & operator-=(const matrix_t& vr);

    // info
    size_t rows() const;
    size_t cols() const;
    
    
    // accessors
    double *  row(const int i) const;
    double *  operator[](const int i) const;
    double *operator[](const size_t i) const;
    void setRaw(double ** raw, const size_t & rows, const size_t & cols);
    double ** toArray() const;

    
    // initializations
    void resize(const size_t &n, const size_t &m);
    void reset(const size_t &n, const size_t &m, const double &val);
    void fill(const double &val);
    void setIdentity(const size_t &n, const size_t &m);
    
    
    // math functions
    vec_t lowerProduct(const vec_t & v) const;
    vec_t diagProduct(const vec_t & v) const;
    vec_t product(const vec_t & v) const;
    void multiply(const double & d);
    double determinantDiag() const;

    matrix_t transpose() const;

    // diagonal matrices
    void diagonalMatrixSquareRoot();

  private:
    
    // the raw data is private, although accessible via toArray();
    double **raw;
    size_t raw_rows;
    size_t raw_cols;
    void free_raw();
    
  };
  
  matrix_t operator+(const matrix_t &ml, const matrix_t &mr);
  matrix_t operator-(const matrix_t &ml, const matrix_t &mr);

  matrix_t operator*(const matrix_t &ml, const matrix_t &mr);
  vec_t operator*(const matrix_t &m, const vec_t &v);
  
  
  std::ostream &operator<<(std::ostream &os, matrix_t const &m);
  
}
