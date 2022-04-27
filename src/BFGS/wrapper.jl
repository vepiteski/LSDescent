
function (obj::L_BFGS_B)(nlp :: AbstractNLPModel;
                         #x₀::Array;
                         #btype = [],
                         mem ::Int64 = 5,
                         stp :: AbstractStopping = NLPStopping(nlp,
                                                               NLPAtX(x₀)
                                                               ),
                         factr::Float64 = 1e1,
                         iprint::Int64 = -1 # does not print
                         )

    stp.meta.optimality_check = optim_check_bounded

    
    x = copy(nlp.meta.x0)
    m = mem
    
    n = length(x)
    f = 0.0
    # clean up
    fill!(obj.task, Cuchar(' '))
    fill!(obj.csave, Cuchar(' '))
    fill!(obj.lsave, zero(Cint))
    fill!(obj.isave, zero(Cint))
    fill!(obj.dsave, zero(Cdouble))
    fill!(obj.wa, zero(Cdouble))
    fill!(obj.iwa, zero(Cint))
    fill!(obj.g, zero(Cdouble))
    fill!(obj.nbd, zero(Cint))
    fill!(obj.l, zero(Cdouble))
    fill!(obj.u, zero(Cdouble))

    # set bounds
    for i = 1:n
        obj.l[i] = nlp.meta.lvar[i]
        obj.u[i] = nlp.meta.uvar[i]
        if obj.l[i] > -Inf
            if obj.u[i] < Inf
                obj.nbd[i] = 2
            else
                obj.nbd[i] = 1
            end
        elseif obj.u[i] < Inf
            obj.nbd[i] = 3
        else
            obj.nbd[i] = 0
        end
    end


    # start

    OK = update_and_start!(stp)


    pgtol = [convert(Float64,max(stp.meta.atol, stp.meta.rtol * stp.meta.optimality0))];
    g = copy(obj.g[1:n])
    obj.task[1:5] = b"START"

    while true
        setulb(n, m, x, obj.l, obj.u, obj.nbd, f, obj.g, factr, pgtol, obj.wa,
               obj.iwa, obj.task, iprint, obj.csave, obj.lsave, obj.isave, obj.dsave)
        #@show obj.task
        if obj.task[1:2] == b"FG"
            f, g = objgrad!(nlp, x, g)
            obj.g[1:n] = g
            #stp_part = update_and_stop!(stp, x=x, fx=f, gx=g)
            #if stp_part 
            #    break;
            #end
        elseif obj.task[1:5] == b"NEW_X"
            #@info ["newx", norm(x), f, norm(g, Inf)]
            stp_part = update_and_stop!(stp, x=x, fx=f, gx=g)
            if stp_part  
                break;
            end
        elseif obj.task[1:2] == b"AB"
            status = "abnormal";
            @show status
            break;
        elseif obj.task[1] == b"E"
            status = "error";
            break;
        end
    end
    # Check the stopping tolerances
    sf = stop!(stp)


    return stp
end
