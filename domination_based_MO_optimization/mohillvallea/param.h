#pragma once

/*
 
 HICAM Multi-objective
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "hicam_internal.h"


// start the eda namespace
namespace hicam
{
  
  // Column vectors
  //----------------------------------------------------------------------------
  class vec_t : public std::vector<double> {
    
  public:
    
    // constructors
    vec_t();
    vec_t(const size_t & n);
    vec_t(const size_t & n, const double & val);
    vec_t(const vec_t & other);
    
    // return as array
    double * toArray();
    void setRaw(double * raw, const size_t & size);

    
    // initializations
    void fill(const double & val);
    void reset(const size_t & size, const double & val);
    vec_t & operator=(const vec_t& vr);
    vec_t & operator+=(const vec_t& vr);
    vec_t & operator-=(const vec_t& vr);
    vec_t & operator+=(const double& val);
    vec_t & operator-=(const double& val);
    vec_t & operator*=(const double& val);
    vec_t & operator/=(const double& val);
    
    // math functions
    double dot(const vec_t & v2);
    double norm() const;          // Euclid norm, equla to p.norm() = sqrt(p.squaredNorm());
    double squaredNorm() const;   //
    double infinitynorm() const;  //
    double variance() const; // variance of the vector. 
    double mean() const; // mean of the vector
    double sum() const; // sum of the vector;
    double prod() const; // product of the vector entries;
    double max() const;
    double min() const;
    double max(size_t & index) const;
    double min(size_t & index) const;
    double percentile(const double percent) const;
    void roundDigits(const size_t number_of_digits);
    vec_t abs() const; // returns copy of vector with absolute values;
    double scaled_euclidean_distance(const vec_t & other, const vec_t & weights) const; 
    int nearest_other(const std::vector<vec_t> & other_vecs) const;
    
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
    vec_t get_diagonal() const;
    void set_diagonal_matrix(vec_t diag);
    
    // diagonal matrices
    void diagonalMatrixSquareRoot();

  private:
    
    // the raw data is private, although accessible via toArray();
    double **raw;
    size_t raw_rows;
    size_t raw_cols;
    void free_raw();
    
  };
  
  // overloaded operators for matrices
  // matrix_t operator-(const matrix_t &ml, const matrix_t &mr);
  // matrix_t operator-(const matrix_t &ml, const double &d);
  // matrix_t operator-(const double &d, const matrix_t &ml);
  // matrix_t operator+(const matrix_t &ml, const matrix_t &mr);
  // matrix_t operator+(const matrix_t &ml, const double &d);
  matrix_t operator+(const matrix_t &ml, const matrix_t &mr);

  matrix_t operator*(const matrix_t &ml, const matrix_t &mr);
  vec_t operator*(const matrix_t &m, const vec_t &v);
  
  
  std::ostream &operator<<(std::ostream &os, matrix_t const &m);
  
}
