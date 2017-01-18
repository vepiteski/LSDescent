function driver3
    
    n = 100 ;
    x = ones(1, n) ; % initial guess
    
    Parm = cg_default ; % initialize default parameter values
    
    % set the initial step in the initial line search to be 1
    Parm.step = 1 ;
    % turn off printing of final statistics
    Parm.PrintFinal = 0 ;
    
    % solve the problem
    [x, status, Stats] = ...
       cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad, Parm) ;

    % only print status and sup-norm of the final gradient
    fprintf (1, 'status: %i gnorm: %i\n', status, Stats.gnorm) ;
    
    function f = myvalue(x)
       f = 0 ;
       n = length(x) ;
       for i=1:n
           t = i^0.5 ;
           f = f + exp(x(i)) - t*x(i) ;
       end
    end
    
    function g = mygrad(x)
        n = length(x) ;
        g = zeros (1, n) ;
        for i=1:n
           t = i^0.5 ;
           g(i) = exp(x(i))-t ;
        end
    end
    
    % first is function value,  second is the gradient vector
    function [f, g] = myvalgrad(x)
        n = length(x) ;
        f = 0 ;
        g = zeros (1, n) ;
        for i=1:n
           t = i^0.5 ;
           ex = exp(x(i)) ;
           f = f + ex - t*x(i) ;
           g(i) = ex - t ;
        end
    end

end
