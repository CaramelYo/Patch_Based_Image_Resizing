/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#include <algorithm>
#include <cmath>
#include "disjoint-set.h"

// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
  float w;
  int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

 /**
 * @brief Segment a graph, a disjoint-set forest representing the segmentation
 * @author Pedro Felzenszwalb, Alexander Freytag
 * @date 27-03-2014 ( dd-mm-yyyy, last updated)
 * 
 * @param[in] i_numVertices: number of vertices in graph
 * @param[in] i_numEdges: number of edges in graph
 * @param[in] edges: array of edges
 * @param[in] c: constant for treshold function
 * 
 * @param[out] u universe (disjoint-set forest representing the segmentation)
 */
universe *segment_graph( const int i_numVertices, 
                         const int i_numEdges, 
                         edge *edges, 
                         const float c
                       )
{ 
  // sort edges by weight
  // note: if weights occure more then once, this might lead to non-deterministic (non-reproducable) behaviour, since
  // "Equivalent elements are not guaranteed to keep their original relative order"
  //  std::sort(edges, edges + i_numEdges);
  
  // adaptation: use stable_sort instead, which will keep the relative position of equal elements :)
  std::stable_sort(edges, edges + i_numEdges);

  // make a disjoint-set forest
  universe *u = new universe(i_numVertices);

  // init thresholds
  float *threshold = new float[i_numVertices];
  for (int i = 0; i < i_numVertices; i++)
  {
    threshold[i] = THRESHOLD(1,c);
  }
  


  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < i_numEdges; i++)
  {
    edge *pedge = &edges[i];
    
    // components conected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b)
    {
      if ( (pedge->w <= threshold[a]) && (pedge->w <= threshold[b]) )
      {
        u->join(a, b);
        a = u->find(a);
        threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
    }
  }
    

  // free up
  delete threshold;
  return u;
}

#endif
