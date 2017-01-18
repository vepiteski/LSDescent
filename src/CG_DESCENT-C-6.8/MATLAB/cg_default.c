/* mex routine generating MATLAB structure for cg_descent parameters */

#include <stdio.h>
#include <limits.h>
#include "mex.h"
#include "../cg_descent.c"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* 55 = current length of parameter structure */
    const char *cg_fnames[55] = {"PrintFinal", "PrintLevel", "PrintParms",
                                 "LBFGS", "memory", "SubCheck", "SubSkip",
                                 "eta0", "eta1", "eta2",
                                 "AWolfe", "AWolfeFac", "Qdecay", "nslow",
                                 "StopRule", "StopFac", "PertRule", "eps",
                                 "egrow", "QuadStep", "QuadCutOff",
                                 "QuadSafe", "UseCubic", "CubicCutOff",
                                 "SmallCost", "debug", "debugtol", "step",
                                 "maxit", "ntries", "ExpandSafe",
                                 "SecantAmp", "RhoGrow", "neps", "nshrink",
                                 "nline", "restart_fac", "feps", "nan_rho",
                                 "nan_decay", "delta", "sigma", "gamma",
                                 "rho", "psi0", "psi_lo", "psi_hi",
                                 "psi1", "psi2", "AdaptiveBeta", "BetaLower",
                                 "theta", "qeps", "qrule", "qrestart"} ;
    mxArray *fout, **out ;
    int ifield ;
    double *pdata ;
    cg_parameter *Parm, ParmStruc ;
    Parm = &ParmStruc ;
    if (nlhs != 1 || nrhs != 0)
    {
        mexErrMsgTxt("Usage: no input and only 1 output\n");
    }

    out = &plhs [0] ;
    cg_default(Parm) ;

    *out = mxCreateStructMatrix(1, 1, 55, cg_fnames) ;

    for (ifield = 0; ifield < 55; ifield ++) /*for each column */
    {
        fout = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL) ;
        pdata = (double *) mxGetData (fout) ;

        switch (ifield)
        {
            case 0: *pdata = Parm->PrintFinal ; break ;
            case 1: *pdata = Parm->PrintLevel ; break ;
            case 2: *pdata = Parm->PrintParms ; break ;
            case 3: *pdata = Parm->LBFGS ; break ;
            case 4: *pdata = Parm->memory ; break ;
            case 5: *pdata = Parm->SubCheck ; break ;
            case 6: *pdata = Parm->SubSkip ; break ;
            case 7: *pdata = Parm->eta0 ; break ;
            case 8: *pdata = Parm->eta1 ; break ;
            case 9: *pdata = Parm->eta2 ; break ;
            case 10: *pdata = Parm->AWolfe ; break ;
            case 11: *pdata = Parm->AWolfeFac ; break ;
            case 12: *pdata = Parm->Qdecay ; break ;
            case 13: *pdata = Parm->nslow ; break ;
            case 14: *pdata = Parm->StopRule ; break ;
            case 15: *pdata = Parm->StopFac ; break ;
            case 16: *pdata = Parm->PertRule ; break ;
            case 17: *pdata = Parm->eps ; break ;
            case 18: *pdata = Parm->egrow ; break ;
            case 19: *pdata = Parm->QuadStep ; break ;
            case 20: *pdata = Parm->QuadCutOff ; break ;
            case 21: *pdata = Parm->QuadSafe ; break ;
            case 22: *pdata = Parm->UseCubic ; break ;
            case 23: *pdata = Parm->CubicCutOff ; break ;
            case 24: *pdata = Parm->SmallCost ; break ;
            case 25: *pdata = Parm->debug ; break ;
            case 26: *pdata = Parm->debugtol ; break ;
            case 27: *pdata = Parm->step ; break ;
            case 28: if ( Parm->maxit == INT_INF ) *pdata = DBL_MAX ;
                     else                          *pdata = Parm->maxit ;
                     break ;
            case 29: *pdata = Parm->ntries ; break ;
            case 30: *pdata = Parm->ExpandSafe ; break ;
            case 31: *pdata = Parm->SecantAmp  ; break ;
            case 32: *pdata = Parm->RhoGrow ; break ;
            case 33: *pdata = Parm->neps ; break ;
            case 34: *pdata = Parm->nshrink  ; break ;
            case 35: *pdata = Parm->nline ; break ;
            case 36: *pdata = Parm->restart_fac ; break ;
            case 37: *pdata = Parm->feps ; break ;
            case 38: *pdata = Parm->nan_rho ; break ;
            case 39: *pdata = Parm->nan_decay ; break ;
            case 40: *pdata = Parm->delta ; break ;
            case 41: *pdata = Parm->sigma ; break ;
            case 42: *pdata = Parm->gamma ; break ;
            case 43: *pdata = Parm->rho ; break ;
            case 44: *pdata = Parm->psi0 ; break ;
            case 45: *pdata = Parm->psi_lo ; break ;
            case 46: *pdata = Parm->psi_hi ; break ;
            case 47: *pdata = Parm->psi1 ; break ;
            case 48: *pdata = Parm->psi2 ; break ;
            case 49: *pdata = Parm->AdaptiveBeta ; break ;
            case 50: *pdata = Parm->BetaLower ; break ;
            case 51: *pdata = Parm->theta ; break ;
            case 52: *pdata = Parm->qeps ; break ;
            case 53: *pdata = Parm->qrule ; break ;
            case 54: *pdata = Parm->qrestart ; break ;
        }
        mxSetFieldByNumber(*out, 0, ifield, fout) ;
    }
}
