/* cg_descent mex function

NOTE: if stats structure changes, need to update GetStat

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "mex.h"
#include "../cg_descent.c"

/* start global variables */
mxArray *cg_myvalue ;
mxArray *cg_mygrad ;
mxArray *cg_myvalgrad ;
/* end global variables */

mxArray *EvaluateFunction (mxArray *f, mxArray *x);
double user_value (double *x, INT n);
void user_grad(double *g, double *x, INT n);
double user_valgrad(double *g, double *x, INT n);
void ReadParm (cg_parameter *Parm, const mxArray *prh);
void GetStat (mxArray **out, cg_stats   *Stat);

mxArray *EvaluateFunction (mxArray *f, mxArray *x)
{
    mxArray *ppFevalRhs[2], *y ;
    ppFevalRhs[0] = f ;
    ppFevalRhs[1] = x ;
    mexCallMATLAB (1, &y, 2, ppFevalRhs, "feval") ;
    return y ;
}

double user_value (double *x, INT n)
{
    mxArray *F, *X ;
    X = mxCreateDoubleMatrix (0,0,mxREAL) ;
    mxFree (mxGetPr (X)) ;
    mxSetPr (X, x) ;
    mxSetN (X, 1) ;
    mxSetM (X, n) ;

    F = EvaluateFunction (cg_myvalue, X) ;

    mxSetPr (X, NULL) ;
    mxFree (mxGetPr (X)) ;
    return mxGetScalar (F) ;
}

void user_grad(double *g, double *x, INT n)
{
    mxArray *G, *X ;
    X = mxCreateDoubleMatrix (0,0,mxREAL) ;
    mxFree (mxGetPr (X)) ;
    mxSetPr (X, x) ;
    mxSetN (X, 1) ;
    mxSetM (X, n) ;

    G = EvaluateFunction (cg_mygrad, X) ;

    memcpy (g, mxGetPr(G), sizeof(double)*n) ;
    mxSetPr (X, NULL) ;
    mxFree (mxGetPr (X)) ;
    return ;
}

double user_valgrad(double *g, double *x, INT n)
{
    mxArray *fg [2], *ppFevalRhs [2], *X ;
    double F ;
    X = mxCreateDoubleMatrix (0,0,mxREAL) ;
    mxFree (mxGetPr (X)) ;
    mxSetPr (X, x) ;
    mxSetN (X, 1) ;
    mxSetM (X, n) ;

    ppFevalRhs [0] = cg_myvalgrad ;
    ppFevalRhs [1] = X ;
    mexCallMATLAB (2, fg, 2, ppFevalRhs, "feval") ;

    F = mxGetScalar (fg[0]) ;   /* first output is the function value */
    memcpy (g, mxGetPr(fg[1]), sizeof(double)*n) ;
    mxSetPr (X, NULL) ;
    mxFree (mxGetPr (X)) ;
    return F ;
}

void ReadParm (cg_parameter *Parm, const mxArray *prh)
{
    double t ;
    Parm->PrintFinal = (int) mxGetScalar(mxGetField(prh, 0, "PrintFinal")) ;
    Parm->PrintLevel = (int) mxGetScalar(mxGetField(prh, 0, "PrintLevel"))  ;
    Parm->PrintParms = (int) mxGetScalar(mxGetField(prh, 0, "PrintParms")) ;
    Parm->LBFGS = (int) mxGetScalar(mxGetField(prh, 0, "LBFGS")) ;
    Parm->memory = (int) mxGetScalar(mxGetField(prh, 0, "memory")) ;
    Parm->SubCheck = (int) mxGetScalar(mxGetField(prh, 0, "SubCheck")) ;
    Parm->SubSkip = (int) mxGetScalar(mxGetField(prh, 0, "SubSkip")) ;
    Parm->eta0 = mxGetScalar(mxGetField(prh, 0, "eta0")) ;
    Parm->eta1 = mxGetScalar(mxGetField(prh, 0, "eta1")) ;
    Parm->eta2 = mxGetScalar(mxGetField(prh, 0, "eta2")) ;
    Parm->AWolfe = (int) mxGetScalar(mxGetField(prh, 0, "AWolfe")) ;
    Parm->AWolfeFac = mxGetScalar(mxGetField(prh, 0, "AWolfeFac"))  ;

    Parm->Qdecay = mxGetScalar(mxGetField(prh, 0, "Qdecay")) ;
    Parm->nslow = (int) mxGetScalar(mxGetField(prh, 0, "nslow")) ;
    Parm->StopRule = (int) mxGetScalar(mxGetField(prh, 0, "StopRule")) ;
    Parm->StopFac = mxGetScalar(mxGetField(prh, 0, "StopFac")) ;
    Parm->PertRule = (int) mxGetScalar(mxGetField(prh, 0, "PertRule")) ;

    Parm->eps = mxGetScalar(mxGetField(prh, 0, "eps")) ;
    Parm->egrow = mxGetScalar(mxGetField(prh, 0, "egrow")) ;
    Parm->QuadStep = (int) mxGetScalar(mxGetField(prh, 0, "QuadStep"))  ;
    Parm->QuadCutOff = mxGetScalar(mxGetField(prh,0, "QuadCutOff")) ;
    Parm->QuadSafe = mxGetScalar(mxGetField(prh, 0, "QuadSafe")) ;

    /* T => when possible, use a cubic step in the line search */
    Parm->UseCubic = (int) mxGetScalar(mxGetField(prh, 0, "UseCubic")) ;
    Parm->CubicCutOff = mxGetScalar(mxGetField(prh,0, "CubicCutOff")) ;
    Parm->SmallCost = mxGetScalar(mxGetField(prh, 0, "SmallCost")) ;
    Parm->debug = (int) mxGetScalar(mxGetField(prh, 0, "debug")) ;
    Parm->debugtol = mxGetScalar(mxGetField(prh, 0, "debugtol")) ;

    Parm->step = mxGetScalar(mxGetField(prh, 0, "step")) ;
    t = mxGetScalar(mxGetField(prh, 0, "maxit")) ;
    if ( t == DBL_MAX ) Parm->maxit = INT_INF ;
    else  Parm->maxit = (INT) mxGetScalar(mxGetField(prh, 0, "maxit")) ;
    Parm->ntries = (int) mxGetScalar(mxGetField(prh, 0, "ntries")) ;
    Parm->ExpandSafe = mxGetScalar(mxGetField(prh, 0, "ExpandSafe")) ;
    Parm->SecantAmp = mxGetScalar(mxGetField(prh,0,"SecantAmp")) ;

    Parm->RhoGrow = mxGetScalar(mxGetField(prh, 0, "RhoGrow")) ;
    Parm->neps = (int) mxGetScalar(mxGetField(prh, 0, "neps")) ;
    Parm->nshrink = (int) mxGetScalar(mxGetField(prh, 0, "nshrink")) ;
    Parm->nline = (int) mxGetScalar(mxGetField(prh, 0, "nline")) ;
    Parm->restart_fac = mxGetScalar(mxGetField(prh, 0, "restart_fac")) ;

    Parm->feps = mxGetScalar(mxGetField(prh, 0, "feps")) ;
    Parm->nan_rho = mxGetScalar(mxGetField(prh, 0, "nan_rho")) ;
    Parm->nan_decay = mxGetScalar(mxGetField(prh, 0, "nan_decay")) ;
    Parm->delta = mxGetScalar(mxGetField(prh, 0, "delta")) ;
    Parm->sigma = mxGetScalar(mxGetField(prh, 0, "sigma")) ;

    Parm->gamma = mxGetScalar(mxGetField(prh, 0, "gamma")) ;
    Parm->rho = mxGetScalar(mxGetField(prh, 0, "rho")) ;
    Parm->psi0 = mxGetScalar(mxGetField(prh, 0, "psi0")) ;
    Parm->psi_lo = mxGetScalar(mxGetField(prh, 0, "psi_lo")) ;
    Parm->psi_hi = mxGetScalar(mxGetField(prh, 0, "psi_hi")) ;

    Parm->psi1 = mxGetScalar(mxGetField(prh, 0, "psi1")) ;
    Parm->psi2 = mxGetScalar(mxGetField(prh, 0, "psi2")) ;
    Parm->AdaptiveBeta = (int) mxGetScalar(mxGetField(prh, 0, "AdaptiveBeta")) ;
    Parm->BetaLower = mxGetScalar(mxGetField(prh, 0, "BetaLower")) ;
    Parm->theta = mxGetScalar(mxGetField(prh, 0, "theta")) ;

    Parm->qeps = mxGetScalar(mxGetField(prh, 0, "qeps")) ;
    Parm->qrestart = (int) mxGetScalar(mxGetField(prh, 0, "qrestart")) ;
    Parm->qrule = mxGetScalar(mxGetField(prh, 0, "qrule")) ;
}

void GetStat (mxArray **out, cg_stats   *Stat)
{
    mxArray *fout ;
    int ifield ;
    double *pdata ;
    const char *cg_fnames[5] = {"f", "gnorm", "iter", "nfunc", "ngrad"} ;
    *out = mxCreateStructMatrix(1, 1, 5, cg_fnames) ;
    for (ifield = 0; ifield < 5; ifield ++)
    {
        fout = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL) ;
        pdata = (double *)mxGetData (fout) ;

        switch (ifield)
        {
            case 0: *pdata = Stat->f ; break ;
            case 1: *pdata = Stat->gnorm ; break ;
            case 2: *pdata = Stat->iter ; break ;
            case 3: *pdata = Stat->nfunc ; break ;
            case 4: *pdata = Stat->ngrad ; break ;
        }
        mxSetFieldByNumber(*out, 0, ifield, fout) ;
    }
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double *x, *Work, tol ; /* input */
    int FoundParm, FoundValgrad, i, mem ;
    INT n ;
    double *newx, *status, status_value ; /* output */
    cg_parameter *Parm, ParmStruc ;
    cg_stats *Stat, StatStruc ;
    Parm = &ParmStruc ;

    if (nrhs < 4 || nrhs > 6)
    {
        mexErrMsgTxt("Only 4, 5 or 6 inputs allowed for cg_descent\n") ;
    }
    else if ( nlhs > 3)  /* there are at most 3 outputs */
        mexErrMsgTxt("More than 3 outputs not allowed for cg_descent\n") ;

    if ( !mxIsClass(prhs[0],"double") )
    {
        mexErrMsgTxt("1st argument of cg_descent must be double\n") ;
    }
    if ( !mxIsClass(prhs[1],"double") )
    {
        mexErrMsgTxt("2nd argument of cg_descent must be double\n") ;
    }
    if ( !mxIsClass(prhs[2],"function_handle") )
    {
        mexErrMsgTxt("3rd argument of cg_descent must be function handle\n") ;
    }
    if ( !mxIsClass(prhs[3],"function_handle") )
    {
        mexErrMsgTxt("4th argument of cg_descent must be function handle\n") ;
    }

    x = mxGetPr (prhs[0]) ;
    n = MAX (mxGetN (prhs[0]), mxGetM (prhs[0])) ;
    tol = mxGetScalar (prhs[1]) ;  
    cg_myvalue = (mxArray*) prhs [2] ;  /* handle for function evaluation */
    cg_mygrad = (mxArray*) prhs [3] ;   /* handle for gradient evaluation */

    FoundValgrad = FALSE ;
    FoundParm = FALSE ;
    for (i = 4; i < nrhs; i++)
    {
        if (mxIsClass(prhs[i],"struct"))
        {
            if ( FoundParm == TRUE )
            {
                mexErrMsgTxt("Too many structures input to cg_descent\n") ;
            }
            ReadParm (Parm, prhs[i]) ;
            FoundParm = TRUE ;
        }
        else if (mxIsClass(prhs[i],"function_handle"))
        {
            if ( FoundValgrad == TRUE )
            {
                mexErrMsgTxt("Too many function handles input to cg_descent\n");
            }
            /* handle for function and gradient evaluation */
            cg_myvalgrad = (mxArray*) prhs [i] ;
            FoundValgrad = TRUE ;
        }
        else
        {
            printf ("Argument %i of cg_descent not understood\n", i) ;
            mexErrMsgTxt ("Stop") ;
        }
    }

    if ( !FoundParm ) cg_default (Parm) ;

    /* allocate memory */
    mem = MIN (Parm->memory, n) ;
    if ( mem == 0 ) /* original CG_DESCENT without memory */
    {
        Work = (double *) mxMalloc (4*n*sizeof (double)) ;
    }
    else if ( Parm->LBFGS || (mem >= n) ) /* use L-BFGS */
    {
        Work = (double *) mxMalloc ((2*mem*(n+1)+4*n)*sizeof (double)) ;
    }
    else /* limited memory CG_DESCENT */
    {
        i = (mem+6)*n + (3*mem+9)*mem + 5 ;
        Work = (double *) mxMalloc (i*sizeof (double)) ;
    }

    plhs [0] = mxDuplicateArray(prhs[0]) ;
    newx = mxGetPr (plhs[0]) ;

    if (nlhs >= 2)
    {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL) ;
        status = mxGetPr(plhs[1]) ;
    }
    else
    {
        status = &status_value ;
    }

    if ( nlhs == 3 )
    {
        Stat = &StatStruc ;
    }
    else Stat = (cg_stats *) NULL ;

    if ( FoundValgrad )
    {
        *status = cg_descent (newx, n, Stat, Parm, tol,
                  user_value, user_grad, user_valgrad, Work) ;
    }
    else
    {
        *status = cg_descent (newx, n, Stat, Parm, tol,
                  user_value, user_grad, NULL, Work) ;
    }
    mxFree (Work) ;

    if (nlhs == 3)
    {
        GetStat (&plhs[2], Stat) ;
    }
    return ;
}
