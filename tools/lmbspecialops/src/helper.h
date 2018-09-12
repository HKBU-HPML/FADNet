//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#ifndef HELPER_H
#define HELPER_H
#include "Eigen/Core"


template <class Derived>
inline bool matrix_is_finite( const Eigen::MatrixBase<Derived>& mat )
{
  for( int j = 0; j < mat.cols(); ++j )
  for( int i = 0; i < mat.rows(); ++i )
  {
    if( !std::isfinite(mat(i,j)) )
      return false;
  }
  return true;

}


inline int divup( int x, int y )
{
  div_t tmp = std::div(x,y);
  return tmp.quot + (tmp.rem != 0 ? 1 : 0);
}


#endif /* HELPER_H */
