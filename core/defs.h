/*---------------------------------------------------------------------------------

  DEFS.H

  -GLOBAL VARIABLE DEFINITIONS

---------------------------------------------------------------------------------*/

#pragma once

// Zone flags
GridInt pflag;
GridInt fail_save;
GridInt fflag;

// Parameters
// physical
double a;
double gam;
double Rhor;
double tp_over_te;

// geometry
double Rin, Rout, hslope;
double poly_norm, poly_xt, poly_alpha, mks_smooth;
double cour;
double dV, dx[NDIM], startx[NDIM];
double x1Min, x1Max, x2Min, x2Max;
double dt, dt_light;
double t, tf;
double rcurr, hcurr;
int istart, istop, jstart, jstop;
int nstep;
int is_restart;


// fluid dumps
double DTd;
double DTf;
double DTl;
int DTr;
int DTp;
int dump_cnt;
double tdump, tlog;

// derived logged output
double mdot, edot, ldot;
double mdot_eh, edot_eh, ldot_eh;
int icurr, jcurr;

// Number of OpenMP threads
int nthreads;

// Electron-heating related globals
#if ELECTRONS
double game, gamp;
double fel0;
double tptemin, tptemax;
#endif
