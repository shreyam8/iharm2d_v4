#include "decs.h"

#if THEORY == EDGB
void edgb_KS_func(double X[NDIM], double edgb_KS[NDIM][NDIM]) ;

void edgb_BL_func(double r, double th, double edgb_BL[NDIM][NDIM]) ; 

void edgb_trans(double r, double th, double edgb_T[NDIM][NDIM]) ; 
#endif