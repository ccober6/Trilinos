/*
 * Copyright(C) 1999-2020, 2023 National Technology & Engineering Solutions
 * of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
 * NTESS, the U.S. Government retains certain rights in this software.
 *
 * See packages/seacas/LICENSE for details
 */

#include "refine_map.h" // for refine_edata
#include "structs.h"
#include <stdio.h> // for NULL

void update_mesh_edata(int                   vertex,     /* graph vertex being worked on */
                       int                   dim,        /* mesh dimension to be adjusted */
                       struct refine_edata  *edata,      /* data structure for edge preferences */
                       struct refine_vdata  *vdata,      /* data structure for vertex preferences */
                       struct vtx_data     **comm_graph, /* communication graph */
                       int                   mesh_dims[3], /* extent of mesh */
                       int                  *node2vtx,     /* maps mesh nodes to comm_graph vtxs */
                       int                  *vtx2node,     /* maps mesh nodes to comm_graph vtxs */
                       double               *best_desire,  /* best desire seen */
                       int                   imax,         /* offset in desire_ptr array */
                       struct refine_edata **desire_ptr    /* buckets for desire values */
)
{
  struct refine_edata *eguy;       /* data for desired edge */
  float                old_desire; /* original desire for edge to flip */
  float                new_desire; /* new desire for edge to flip */
  int                  i, k;       /* loop counter */

  for (i = 0; i < 2; i++) { /* Have to adjust two edges. */
    dim  = -(dim + 1);
    eguy = find_edge_mesh(vertex, dim, edata, mesh_dims, vtx2node);
    if (eguy != NULL) {

      old_desire = eguy->swap_desire;
      new_desire = compute_mesh_edata(eguy, vdata, mesh_dims, comm_graph, node2vtx);

      if (new_desire != old_desire) { /* Update linked list if necessary. */
        eguy->swap_desire = new_desire;

        if (new_desire > *best_desire) {
          *best_desire = new_desire;
        }

        /* Remove eguy from it's current place in list. */
        if (eguy->prev == NULL) {
          /* Round up for index into desire_ptr. */
          if (old_desire >= 0) {
            k = old_desire;
            if (k != old_desire) {
              k++;
            }
          }
          else {
            k = -old_desire;
            if (k != -old_desire) {
              k++;
            }
            k = -k;
          }
          k += imax;
          desire_ptr[k] = eguy->next;
        }
        else {
          eguy->prev->next = eguy->next;
        }
        if (eguy->next != NULL) {
          eguy->next->prev = eguy->prev;
        }

        /* Now add eguy to it's new desire bucket. */
        if (new_desire >= 0) {
          k = new_desire;
          if (k != new_desire) {
            k++;
          }
        }
        else {
          k = -new_desire;
          if (k != -new_desire) {
            k++;
          }
          k = -k;
        }
        k += imax;

        eguy->prev = NULL;
        eguy->next = desire_ptr[k];
        if (desire_ptr[k] != NULL) {
          desire_ptr[k]->prev = eguy;
        }
        desire_ptr[k] = eguy;
      }
    }
  }
}
