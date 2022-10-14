using MPI, Test

function runmpi(file; ntasks = 1)
    # by default some mpi runtimes will
    # complain if more resources (processes)
    # are requested than available on the node
    # set OMPI_MCA_rmaps_base_oversubscribe=1 for openmpi
    @info "Running MPI test..." file ntasks
    @time @test mpiexec() do cmd
        run(`$cmd -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`)
        true
    end
end
