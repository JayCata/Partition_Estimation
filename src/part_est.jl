module MT
include("mt_struct.jl") # Includes all structs and small methods
include("mt_helper.jl") # Includes helper functions
include("mt_grow.jl") # Includes all functions related to growing tree
include("mt_prune.jl") # Includes all functions related to pruning tree
include("mt_join.jl") # Includes all functions related to joining  tree
include("mt_interp.jl") # Includes all functions related to interpreting tree
include("mt_post.jl") # Includes functions that allow for standard error calculations and honest estimation
end