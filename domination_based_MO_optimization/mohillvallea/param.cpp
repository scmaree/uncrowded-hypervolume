

/*
 
 HICAM
 
 By S.C. Maree, 2016
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "param.h"
#include "mathfunctions.h"

namespace hicam
{
  
  // constructors
  vec_t::vec_t() : std::vector<double>() {}
  vec_t::vec_t(const size_t & n) : std::vector<double>(n,0.0) {}
  vec_t::vec_t(const size_t & n, const double & val) : std::vector<double>(n,val) {}
  
  vec_t::vec_t(const vec_t & other)
  {
    (*this) = other;
  }
  
  
  
  // return as an array to use with the LINPACK/BLAS library functions
  double * vec_t::toArray()
  {
    return &(*this)[0];
  }
  
  void vec_t::setRaw(double * raw, const size_t & size)
  {
    
    this->resize(size);
    
    for(size_t i = 0; i < size; ++i) {
      (*this)[i] = raw[i];
    }
    
  }

  
  // math functions, {+,-,/,*}
  //--------------------------------------------------------
  double vec_t::dot(const vec_t & v2) // we cannot use consts because we use toArray
  {
    
    assert(this->size() == v2.size());
    // return vectorDotProduct(this->toArray(), v2.toArray(), (int)this->size());
    
    double result;
    
    result = 0.0;
    for (size_t i = 0; i < v2.size(); i++) {
      result += (*this)[i] * v2[i];
    }
    
    return result ;
    
  }

  
  double vec_t::norm() const
  {
    return sqrt(this->squaredNorm());
  }
  
  double vec_t::squaredNorm() const
  {
    
    double v = 0;
    
    for(size_t i = 0; i < this->size(); ++i) {

      v += (*this)[i] * (*this)[i];
    }
    
    return v;
  }
  
  
  // |x|_infty = max{|x1|,...,|xn|}
  double vec_t::infinitynorm() const
  {
    
    assert(this->size() > 0);
    
    double v = fabs((*this)[0]);
    
    for(size_t i = 1; i < this->size(); ++i) {
      if( fabs( (*this)[i] ) > v )
        v = fabs( (*this)[i] );
    }
    
    return v;
  }

  // mean of the vector
  double vec_t::mean() const
  {
    double mean = 0.0;
    for (size_t i = 0; i < this->size(); ++i)
      mean += (*this)[i];

    mean /= this->size();

    return mean;

  }
  
  // sum of the vector
  double vec_t::sum() const
  {

    double total = 0.0;

    for (size_t i = 0; i < this->size(); ++i)
    {
      total += (*this)[i];
    }

    return total;
  }

  // sum of the vector
  double vec_t::prod() const
  {

    double total = 1.0;

    for (size_t i = 0; i < this->size(); ++i)
    {
      total *= (*this)[i];
    }

    return total;
  }

  // min of the vector
  double vec_t::min() const
  {

    assert(size() > 0);
      
    double min = (*this)[0];

    for (size_t i = 1; i < this->size(); ++i)
    {
      if (min > (*this)[i])
        min = (*this)[i];
    }

    return min;
  }

  // min of the vector
  double vec_t::max() const
  {

    assert(size() > 0);

    double max = (*this)[0];

    for (size_t i = 1; i < this->size(); ++i)
    {
      if (max < (*this)[i])
        max = (*this)[i];
    }

    return max;
  }

  double vec_t::min(size_t & index) const
  {
    
    assert(size() > 0);
    
    double min = (*this)[0];
    index = 0;
    for (size_t i = 1; i < this->size(); ++i)
    {
      if (min > (*this)[i]) {
        min = (*this)[i];
        index = i;
      }
    }
    
    return min;
  }
  
  // min of the vector
  double vec_t::max(size_t & index) const
  {
    
    assert(size() > 0);
    
    double max = (*this)[0];
    index = 0;
    for (size_t i = 1; i < this->size(); ++i)
    {
      if (max < (*this)[i]) {
        max = (*this)[i];
        index = i;
      }
    }
    
    return max;
  }
  
  // variance of the vector. 
  double vec_t::variance() const
  {

    double mean = this->mean();
    double variance = 0.0;
    double diff = 0.0;

    for (size_t i = 0; i < this->size(); ++i)
    {
      diff = (*this)[i] - mean;
      variance += diff*diff;
    }

    variance /= this->size();


    return variance;
  }

  double vec_t::percentile(const double percent) const
  {
    
    vec_t this_copy = (*this);

    std::sort(this_copy.begin(), this_copy.end());

    size_t i = (size_t)floor(percent / 100.0 * this_copy.size());
    size_t j = (size_t)std::min(this->size()-1.0, ceil(percent / 100.0 * this_copy.size()));

    // interpolate the result
    return this_copy[i] + (this_copy[j] - this_copy[i])*(percent / 100.0 * this_copy.size() - floor(percent / 100.0 * this_copy.size()));

  }

  void vec_t::roundDigits(const size_t number_of_digits)
  {

    double presicion = pow(10.0, (double)number_of_digits);

    for (size_t i = 0; i < this->size(); ++i)
    {
      (*this)[i] = round((*this)[i] * presicion) / presicion;
    }
      

  }
  
  
  int vec_t::nearest_other(const std::vector<vec_t> & other_vecs) const
  {
    
    double nearest_distance = 1e300;
    double current_distance;
    int nearest_vec = -1;
    
    for (int j = 0; j < (int) other_vecs.size(); ++j)
    {
      current_distance = (*this - other_vecs[j]).norm();
      
      if (current_distance < nearest_distance)
      {
        nearest_distance = current_distance;
        nearest_vec = j;
      }
    }
    
    return nearest_vec;
  }

  

  vec_t vec_t::abs() const
  {
    vec_t vabs(this->size());

    for (size_t i = 0; i < this->size(); ++i)
    {
      vabs[i] = std::abs((*this)[i]);
    }

    return vabs;
  }
  
  double vec_t::scaled_euclidean_distance(const vec_t & other, const vec_t & weights) const
  {
    vec_t distance(this->size());
    for (size_t i = 0; i < this->size(); ++i)
    {
      distance[i] = ((*this)[i] - other[i]) / weights[i];
    }

    return distance.norm();
  }
  
  void vec_t::fill(const double & val)
  {
    
    for(size_t i = 0; i < this->size(); ++i) {
      (*this)[i] = val;
    }
    
  }
  
  void vec_t::reset(const size_t & size, const double & val)
  {
     resize(size);
     fill(val);
  }

  
  // param = param - param
  vec_t operator-(const vec_t &pl, const vec_t &pr)
  {
    
    assert (pl.size() == pr.size());
    
    vec_t r(pl.size(),0.0);
    
    for(size_t i = 0; i < pl.size(); i++) {
      r[i] = pl[i] - pr[i];
    }
    
    return r;
    
  }
  
  // vec = vec - double
  vec_t operator-(const vec_t &p, const double &v)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = p[i] - v;
    }
    
    return r;
    
  }
  
  // vec = double - vec
  vec_t operator-(const double &v, const vec_t &p)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = v - p[i];
    }
    
    return r;
    
  }
  
  
  // param = param + param
  vec_t operator+(const vec_t &pl, const vec_t &pr)
  {
    
    assert (pl.size() == pr.size());
    
    vec_t r(pl.size(),0.0);
    
    for(size_t i = 0; i < pl.size(); i++) {
      r[i] = pl[i] + pr[i];
    }
    
    return r;
    
  }
  
  // param = param + double
  vec_t operator+(const vec_t &p, const double &v)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = p[i] + v;
    }
    
    return r;
    
  }
  
  // param = double + param
  vec_t operator+(const double &v, const vec_t &p)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = v + p[i];
    }
    
    return r;
    
  }
  
  // param = double * param;
  vec_t operator*(const double &v, const vec_t &p)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = v * p[i];
    }
    
    return r;
    
  }
  
  // param = param * double
  vec_t operator*(const vec_t &p, const double &v)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = p[i] * v;
    }
    
    return r;
    
  }
  
  // param = param / double
  vec_t operator/(const vec_t &p, const double &v)
  {
    
    vec_t r(p.size(),0.0);
    
    for(size_t i = 0; i < p.size(); i++) {
      r[i] = p[i] / v;
    }
    
    return r;
    
  }
  
  
  // Assignment operators
  //----------------------------------------------------------------------
  vec_t & vec_t::operator=(const vec_t& pr) {

    resize(pr.size());

    for (size_t i = 0; i < pr.size(); ++i) {
      (*this)[i] = pr[i];
    }

    return *this;

  }

  
  vec_t & vec_t::operator+=(const vec_t& pr) {
    
    assert(this->size() == pr.size());
    
    for(size_t i = 0; i < pr.size(); ++i) {
      (*this)[i] += pr[i];
    }
    
    return *this;
    
  }
  
  vec_t & vec_t::operator-=(const vec_t& pr) {
    
    assert(this->size() == pr.size());
    
    for(size_t i = 0; i < pr.size(); ++i) {
      (*this)[i] -= pr[i];
    }
    
    return *this;
    
  }
  
  vec_t & vec_t::operator+=(const double& v) {
    
    for(size_t i = 0; i < this->size(); ++i) {
      (*this)[i] += v;
    }
    
    return *this;
    
  }
  
  vec_t & vec_t::operator-=(const double& v) {
    
    for(size_t i = 0; i < this->size(); ++i) {
      (*this)[i] -= v;
    }
    
    return *this;
    
  }
  
  vec_t & vec_t::operator*=(const double& v) {
    
    for(size_t i = 0; i < this->size(); ++i) {
      (*this)[i] *= v;
    }
    
    return *this;
    
  }
  
  vec_t & vec_t::operator/=(const double& v) {
    
    for(size_t i = 0; i < this->size(); ++i) {
      (*this)[i] /= v;
    }
    
    return *this;
    
  }
  
  
  
  
  // stream output overloaded
  std::ostream &operator<<(std::ostream &os, vec_t const &p)  {
    
    for(size_t i = 0; i < p.size(); ++i) {
      os << p[i] << " ";
    }
    
    return os;
  }
  
  
  
  // matrices
  //----------------------------------------------------------------------------
  // constructors
  matrix_t::matrix_t() {
    raw = NULL;
    raw_rows = 0;
    raw_cols = 0;
  }
  
  matrix_t::matrix_t(const size_t &n, const size_t &m) {
    raw = NULL;
    raw_rows = 0;
    raw_cols = 0;
    resize(n,m);
  }
  
  matrix_t::matrix_t(const size_t &n, const size_t &m, const double &v) {
    raw = NULL; 
    raw_rows = 0;
    raw_cols = 0;
    resize(n,m);
    fill(v);
  }
  
  // copy constructor
  matrix_t::matrix_t(const matrix_t &m)
  {
    raw = NULL;
    raw_rows = m.raw_rows;
    raw_cols = m.raw_cols;
    raw = matrixNew((int)raw_rows, (int)raw_cols);

    for (size_t i = 0; i < rows(); ++i)
      for (size_t j = 0; j < cols(); ++j)
        raw[i][j] = m[i][j];

  }


  matrix_t & matrix_t::operator=(const matrix_t &m)
  {

    free_raw();

    raw_rows = m.raw_rows;
    raw_cols = m.raw_cols;
    raw = matrixNew((int)raw_rows, (int)raw_cols);

    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j) {
        raw[i][j] = m[i][j];
      }
    }

    return *this;
  }

  matrix_t & matrix_t::operator+=(const matrix_t & m)
  {

    assert(m.rows() == rows());
    assert(m.cols() == cols());

    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j)
      {
        raw[i][j] += m[i][j];
      }
    }

    return *this;

  }


  // destructor
  matrix_t::~matrix_t() {
    free_raw();
  }
  
  // free the raw data
  void matrix_t::free_raw()
  {
    if (raw!=NULL)
    {
      for (size_t i = 0; i < rows(); ++i) {
        free(raw[i]);
      }

      free(raw);
      
      raw = NULL;
      raw_rows = 0;
      raw_cols = 0;
    }
  }
  
  
  // clears the matrix
  void matrix_t::resize(const size_t &n, const size_t &m)
  {
    free_raw();
    raw = matrixNew( (int)n, (int)m );
    raw_rows = n;
    raw_cols = m;
  }
  
  
  // multiply the matrix with a constant
  void matrix_t::multiply(const double & d)
  {
    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j) {
        raw[i][j] *= d;
      }
    }
  }
  
  double matrix_t::determinantDiag() const
  {
    return get_diagonal().prod();
  }

  void matrix_t::diagonalMatrixSquareRoot()
  {

    for (size_t i = 0; i < rows(); i++) {
      for (size_t j = 0; j < cols(); j++)
      {
        if (i != j) {
          raw[i][j] = 0.0;
        }
        else {
          raw[i][j] = sqrt(raw[i][j]);
        }
      }
    }

  }

  matrix_t matrix_t::transpose() const
  {
    matrix_t t(rows(),cols());

    for (size_t i = 0; i < rows(); i++) {
      for (size_t j = 0; j < cols(); j++)
      {
        t[i][j] = raw[j][i];
      }
    }

    return t;
  }
  
  vec_t matrix_t::get_diagonal() const
  {
    
    vec_t diag(rows(), 0.0);
    
    for(size_t i = 0; i < rows(); ++i)
    {
      diag[i] = raw[i][i];
    }
    
    return diag;
  }
  
  
  void matrix_t::set_diagonal_matrix(vec_t diag)
  {
    reset(diag.size(), diag.size(), 0.0);
    
    for(size_t i = 0; i < diag.size(); ++i) {
      raw[i][i] = diag[i];
    }
    
  }
  
  // access operators
  //------------------------------------------------------------
  double * matrix_t::row(const int i) const
  {
    
    return raw[i];
    
  }

  
  double * matrix_t::operator[](const int i) const
  {
    assert(i < (int) raw_rows);
    return row(i);
    
  }

  
  double * matrix_t::operator[](const size_t i) const
  {
    assert(i < raw_rows);
    return row((int)i);
    
  }

  
  double ** matrix_t::toArray() const
  {
    return raw;
  }

  
  void matrix_t::setRaw(double ** raw, const size_t & rows, const size_t & cols)
  {
    
    free_raw();

    this->raw = raw;
    raw_rows = rows;
    raw_cols = cols;
    
  }
  
  
  size_t matrix_t::rows() const
  {
    return raw_rows;
  }
  
  size_t matrix_t::cols() const
  {
    return raw_cols;
  }
  
  // initializations
  // scm: if time, improve this by not completely free-ing the memorty
  void matrix_t::reset(const size_t &n, const size_t &m, const double &v)
  {
    resize(n,m);
    fill(v);
  }
  
  void matrix_t::fill(const double &v)
  {
    for(size_t i = 0; i < raw_rows; ++i) {
      for(size_t j = 0; j < raw_cols; ++j) {
        raw[i][j] = v;
      }
    }
  }
  
  void matrix_t::setIdentity(const size_t &n, const size_t &m)
  {
    this->reset(n,m, 0.0);
    
    for(size_t i = 0; i < std::min(n,m); ++i) {
      raw[i][i] = 1.0;
    }
  }
  
  
  // Matrix-matrix product
  // this copies a matrix...
  matrix_t operator*(const matrix_t &ml, const matrix_t &mr)
  {
    
    assert(ml.cols() == mr.rows());
    
    int n0 = (int)ml.rows();
    int n1 = (int)ml.cols();
    int n2 = (int)mr.cols();
   
    matrix_t result(n0,n2,0.0);
    
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        for (int k = 0; k < n2; ++k) {
          result[i][j] += ml[i][k] * mr[k][j];
        }
      }
    }

    // result.setRaw(matrixMatrixMultiplication(ml.toArray(),mr.toArray(), n0, n1, n2),n0,n2);

    return result;
    
  }

  // matrix-vector product
  vec_t operator*(const matrix_t &m, const vec_t &v)
  {
    
    assert(m.cols() == v.size());
    
    int n0 = (int)m.rows();
    int n1 = (int)m.cols();
    
    vec_t result(n0, 0.0);

    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        result[i] += m[i][j] * v[j];
      }
    }

    return result;
    
  }
  
  // Matrix-matrix sum
  // this copies a matrix...
  matrix_t operator+(const matrix_t &ml, const matrix_t &mr)
  {

    assert(ml.cols() == mr.rows());

    // int n0 = (int)ml.rows();
    // int n1 = (int)ml.cols();
    // int n2 = (int)mr.cols();

    matrix_t result = ml;

    for (size_t i = 0; i < ml.rows(); ++i) {
      for (size_t j = 0; j < ml.cols(); ++j) {
        result[i][j] += mr[i][j];
      }
    }

    return result;

  }

  
  // stream output overloaded
  std::ostream &operator<<(std::ostream &os, matrix_t const &m)  {
    
    for(size_t i = 0; i < m.rows(); ++i) {
      for(size_t j = 0; j < m.cols(); ++j)
        os << m[i][j] << " ";
      os << std::endl;
    }
    
    return os;
  }

  // only for triangular matrices
  vec_t matrix_t::diagProduct(const vec_t & v) const
  {

    assert(rows() == cols());
    assert(v.size() == rows());

    vec_t result(rows(), 0.0);

    for (size_t i = 0; i < rows(); ++i) {
        result[i] += raw[i][i] * v[i];
    }

    return result;
  }
  
  // only for triangular matrices
  vec_t matrix_t::lowerProduct(const vec_t & v) const
  {

    assert(rows() == cols());
    assert(v.size() == rows());

    vec_t result(rows(), 0.0);

    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j <= i; ++j) {
        result[i] += raw[i][j] * v[j];
      }
    }

    return result;
  }

  // only for triangular matrices
  vec_t matrix_t::product(const vec_t & v) const
  {

    assert(rows() == cols());
    assert(v.size() == rows());

    vec_t result(rows(), 0.0);

    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j) {
        result[i] += raw[i][j] * v[j];
      }
    }

    return result;
  }


  
}

































