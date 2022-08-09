export allocate_cache, run!

# Algorithm interface

"""
    allocate_cache(alg, prototypes...)

Allocate a cache that the algorithm will use during `run!`, given some values
that are `similar` to values that need to be cached.
"""
function allocate_cache(alg, prototypes...) end

"""
    run!(alg, cache, args...)

Run the algorithm for the given arguments, using the `cache` to store
intermediate values. Depending on the algorithm, the arguments may be  modified.
"""
function run!(alg, cache, args...) end
