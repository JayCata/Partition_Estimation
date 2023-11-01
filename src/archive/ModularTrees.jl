module MT
# Libraries needed for pacakge
using Statistics
using StatsBase
using LinearAlgebra
using Random
import Base.show

### Supertypes for Node Level Estimators ###
# NodeEstimator Structures and Methods
abstract type  NodeEstimator
end

# Structure for Gradient Estimator
abstract type GradEstimator <: NodeEstimator
end


 
# OLS Node Estimator
# NodeEstimators requires two functions: fit! and predict.
mutable struct ols <: NodeEstimator
    coeff       :: Vector{Float64}
    ols() = new()
end

function fit!(ne::ols, X, Y)
    ne.coeff = inv(X * X') * (X * Y)[:,1]
end

function predict(ne::ols, X)
    return X' * ne.coeff
end


# Gradient Estimator using OLS
# GradEstimators require four functions: fit!, fit_grad!, predict, and predict_grad 
mutable struct ols_grad <: GradEstimator
    coeff       :: Vector{Float64}
    X_demean    :: Union{Float64, Vector{Float64}}
    Y_demean    :: Float64
    ols_grad() = new()
end

function fit!(ne::ols_grad, X, Y)
    ne.coeff = inv(X * X') * (X * Y)[:,1]
end

function fit_grad!(ne::ols_grad, X, Y)
    ne.X_demean = mean(X, dims = 2)[:, 1]
    ne.Y_demean = mean(Y)
    ne.coeff = inv(X * X') * (X * Y)[:,1] 
end

function predict(ne::ols_grad, X)
    
    return X' * ne.coeff
end

function predict_grad(ne::ols_grad, X)
    return (X'.- ne.X_demean) * ne.coeff .+ ne.Y_demean
end

### FitMeasure Structures and Methods ###
abstract type FitMeasure
end

# FitMeasure requires three functions: fm_calc, fm_compare, fm_aggregate
# The first gives us the measure of fit, the second compares splits, the third aggregates in cross validation

#SSE fm
mutable struct sse <: FitMeasure
    mof         :: Float64
    mof_gain    :: Union{Nothing, Float64}
    sse() = new()
end

function fm_calc_mof!(fm::sse, Y, Y_hat)
    fm.mof = sum((Y - Y_hat).^2)
end

function fm_get_worst_mof!(fm::sse)
    fm.mof = Inf64
end

function fm_compare_grow(l_fm::sse, r_fm::sse, best_l_fm::sse, best_r_fm::sse)
    return l_fm.mof + r_fm.mof < best_l_fm.mof + best_r_fm.mof
end

function fm_compare_prune(new_val, fm_obj::sse)
    if new_val < fm_obj.mof
        return true
    end
    return false
end

function fm_full_tree_reg_val(fm_obj::sse)
    return 0.0
end


# Threshold Objects
mutable struct Threshold
    above       :: Int64
    below       :: Int64
    thresh_var  :: Int64
    thresh_val  :: Float64
    l_endpoints :: Vector{Vector{Float64}}
    r_endpoints :: Vector{Vector{Float64}}
end

# Tree Structures
mutable struct SplitObj
    feature     :: Int64
    min         :: Float64
    max         :: Float64
    split_val   :: Float64
    left_idx    :: Vector{Int64}
    right_idx   :: Vector{Int64}
    left_ne     :: NodeEstimator
    right_ne    :: NodeEstimator
    left_fm     :: FitMeasure
    right_fm    :: FitMeasure
    SplitObj() = new() 
end

mutable struct Node
    feature     :: Int64
    comp_value  :: Float64
    data_idx    :: Vector{Int64}
    left_child  :: Union{Nothing, Node}
    right_child :: Union{Nothing, Node}
    parent      :: Union{Nothing, Node}
    ne          :: NodeEstimator
    locate      :: String
    fm          :: FitMeasure
    Node() = new()
end

mutable struct Tree
    root        :: Node
    depth       :: Int64
    leaves      :: Int64
    comp        :: Vector{String}
    mof_gain    :: Vector{Any}
    regions     :: Int64
    ne_type     :: DataType
    fm_type     :: DataType
    Tree() = new()
end

# Change how nodes, trees, and thresholds are displayed
Base.show(io::IO, f::Tree) = show(io, "$(f.ne_type) Tree of depth $(f.depth) with $(f.leaves) leaves and $(f.regions) regions")
Base.show(io::IO, f::Node) = show(io, "Node split on feature $(f.feature) at value $(f.comp_value)")
Base.show(io::IO, f::Threshold) = show(io, "Region $(f.above) above Region $(f.below) at Q$(f.thresh_var) = $(f.thresh_val) from $(f.l_endpoints) to $(f.r_endpoints).")

# Helper Functions for grow_tree
function create_split_obj(feature, min_val, max_val, split_val, left_idx, right_idx, left_ne, right_ne, left_fm, right_fm)
    new_so = SplitObj()
    new_so.feature = feature
    new_so.min = min_val
    new_so.max = max_val
    new_so.split_val = split_val
    new_so.left_idx = left_idx
    new_so.right_idx = right_idx
    new_so.left_ne = left_ne
    new_so.right_ne = right_ne
    new_so.left_fm = left_fm
    new_so.right_fm = right_fm
    return new_so
end

function bool_to_int_index(a)
    a == 1
end

function find_best_split(X, Y, Q, feature_idx, minobs, ne_type, fm_type)
    # Initialize best variables
    best_feature_idx = -1
    best_split_val = -Inf64
    best_min_val = -Inf64
    best_max_val = Inf64
    best_lidx = nothing
    best_ridx = nothing
    best_l_ne = ne_type()
    best_r_ne = ne_type()
    best_l_fm = fm_type()
    best_r_fm = fm_type()
    fm_get_worst_mof!(best_l_fm)
    fm_get_worst_mof!(best_r_fm)

    # Initialize node estimators
    l_ne = ne_type()
    r_ne = ne_type()
    l_fm = fm_type()
    r_fm = fm_type()

    # Get number of split vars and the sample size
    nfeatures = length(feature_idx)

    if length(Y) >= 2 * minobs
        for i = 1:nfeatures
           
            if nfeatures !== 1
                fct_vals = Q[i,:]
            else
                fct_vals = reshape(Q,length(Q))
            end
            min_val = minimum(fct_vals)
            max_val = maximum(fct_vals)
        
            if min_val !== max_val
                
                feat_uniq = sort(unique(fct_vals))
                feat_uniq = feat_uniq[1:end-1] + diff(feat_uniq)/2
                feat_uniq = shuffle(feat_uniq) 
                for split_val in feat_uniq
                    # Get indices for each side of split
                    lidx = (fct_vals .<= split_val)
                    ridx = (fct_vals .> split_val)
                    
                    # ensure we have at least minobs points
                    if sum(ridx) >= minobs && sum(lidx) >= minobs
                        # Catch singular matrices
                        try 
                            fit!(l_ne, X[:,lidx], Y[lidx])
                        catch err
                            if isa(err, SingularException)
                                #print(" Error with Split Val ", split_val)
                                @goto nextsplitval
                            elseif isa(err, LAPACKException)
                                #print(" Error with Split Val ", split_val)
                                @goto nextsplitval
                            end
                        end
                        try 
                            fit!(r_ne, X[:,ridx], Y[ridx])
                        catch err
                            if isa(err, SingularException)
                                #print(" Error with Split Val ", split_val)
                                @goto nextsplitval
                            elseif isa(err, LAPACKException)
                                #print(" Error with Split Val ", split_val)
                                @goto nextsplitval
                            end
                        end
                
         
                        Y_hat_l = predict(l_ne, X[:,lidx])
                        Y_hat_r = predict(r_ne, X[:,ridx])
                        fm_calc_mof!(l_fm, Y[lidx], Y_hat_l)
                        fm_calc_mof!(r_fm, Y[ridx], Y_hat_r)
                        
                        # Check to see if best split
                        if fm_compare_grow(l_fm, r_fm, best_l_fm, best_r_fm)
                            best_feature_idx = i
                            best_split_val = split_val
                            best_l_fm = deepcopy(l_fm)
                            best_r_fm = deepcopy(r_fm)
                            best_min_val = min_val
                            best_max_val = max_val
                            best_lidx = lidx
                            best_ridx = ridx
                            best_l_ne = deepcopy(l_ne)
                            best_r_ne = deepcopy(r_ne)
                        end
                    end
                    @label nextsplitval
                end
            end

        end
    end
    split_obj = SplitObj()
    split_obj.split_val = best_split_val
    split_obj.min = best_min_val
    split_obj.max = best_max_val
    split_obj.feature = best_feature_idx
    split_obj.left_fm = best_l_fm
    split_obj.right_fm = best_r_fm
    
    if !isinf(best_split_val)
        split_obj.feature = feature_idx[best_feature_idx]
        best_lidx = findall(bool_to_int_index, best_lidx)
        best_ridx = findall(bool_to_int_index, best_ridx)
        split_obj = create_split_obj(feature_idx[best_feature_idx], best_min_val, best_max_val, best_split_val, best_lidx, best_ridx, best_l_ne, best_r_ne, best_l_fm, best_r_fm)
        return split_obj
    end
    return split_obj
end

function queue_compute_nodes!(queue::Vector{Node}, node::Node)
    push!(queue, node)
end

function compute_node!(tree::Tree, node::Node, X, Y, Q, feature_idx, minobs, ne_type, fm_type)
    # last three params if split needs to be restarted
    @views split_obj = find_best_split(X[:,node.data_idx], Y[node.data_idx], Q[:,node.data_idx], feature_idx, minobs, ne_type, fm_type)
    
    node.feature = split_obj.feature
    node.comp_value = split_obj.split_val
    node.fm.mof_gain = nothing 
    node.left_child = nothing
    node.right_child = nothing

    if !isinf(split_obj.split_val)
        left_node = Node()
        right_node = Node()
        left_node.data_idx = node.data_idx[split_obj.left_idx]
        right_node.data_idx = node.data_idx[split_obj.right_idx]
        left_node.ne = split_obj.left_ne
        right_node.ne = split_obj.right_ne
        left_node.feature = -1
        right_node.feature = -1
        left_node.left_child = nothing
        left_node.right_child = nothing
        right_node.left_child = nothing
        right_node.right_child = nothing
        left_node.comp_value = NaN
        right_node.comp_value = NaN
        left_node.locate = string(node.locate, '0')
        right_node.locate = string(node.locate, '1')
        left_node.parent = node
        right_node.parent = node
        left_node.fm = split_obj.left_fm
        right_node.fm = split_obj.right_fm
        node.left_child = left_node
        node.right_child = right_node
        node.fm.mof_gain = node.fm.mof - left_node.fm.mof - right_node.fm.mof # this line could be generalized 
        return left_node, right_node, node.fm.mof_gain
    end
  
    return nothing, nothing, node.fm.mof_gain
end

function create_root_node!(X, Y, Q, feature_idx, minobs, ne_type, fm_type)
    split_obj = find_best_split(X, Y, Q, feature_idx, minobs, ne_type, fm_type)
    
    queue = Vector{Node}()
    if !isinf(split_obj.split_val)
        left_node = Node()
        left_node.feature = -1
        left_node.left_child = nothing
        left_node.right_child = nothing
        left_node.ne= split_obj.left_ne
        left_node.comp_value = NaN
        left_node.data_idx = split_obj.left_idx
        left_node.locate = "0"
        left_node.fm = split_obj.left_fm
        
        right_node = Node()
        right_node.feature = -1
        right_node.left_child = nothing
        right_node.right_child = nothing
        right_node.ne = split_obj.right_ne
        right_node.comp_value = NaN
        right_node.data_idx = split_obj.right_idx
        right_node.locate = "1"
        right_node.fm = split_obj.right_fm
        
        node = Node()
        node.feature = split_obj.feature
        node.comp_value = split_obj.split_val
        node.data_idx = collect(1:size(X)[2])
        node.ne = ne_type()
        node.fm = fm_type()
        fit!(node.ne, X, Y)
        Y_hat = predict(node.ne, X)
        fm_calc_mof!(node.fm, Y, Y_hat)
        node.fm.mof_gain = node.fm.mof - left_node.fm.mof - right_node.fm.mof
        node.left_child = left_node
        node.right_child = right_node
        node.locate = ""
        node.parent = nothing
        
        left_node.parent = node
        right_node.parent = node
        
        queue_compute_nodes!(queue, left_node)
        queue_compute_nodes!(queue, right_node)
        
        return node, queue
    end
    
    node = Node()
    node.fm = fm_type()
    node.fm.mof_gain = nothing
    node.data_idx = [Int(i) for i in (range(1, length(Y), step = 1))]
    node.left_child = nothing
    node.right_child = nothing
    node.ne = ne_type()
    fit!(node.ne, X, Y)
    return node, queue
end

function count_leaves(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 1) 
    end
    return sum(term)
end

# Tree Growing Algorithm
function grow_tree(X, Y, Q, feature_idx, minobs, ne_type, fm_type)
    comp = [""]
    s_time = time()
    if ndims(Q) == 1
        Q = reshape(Q, 1, length(Q))
    end
    tree = Tree()
    depth = 0 
    @views X_view = X
    root, queue = create_root_node!(X_view, Y, Q,  feature_idx, minobs, ne_type, fm_type)
    if length(queue) > 0
        comp = vcat(comp, "0", "1")
    end
    
    mof_gain_list = [root.fm.mof_gain]
    
    while length(queue) > 0
        node = popfirst!(queue)
        left_node, right_node, mof_gain = compute_node!(tree, node, X_view, Y, Q, feature_idx, minobs, ne_type, fm_type)
        mof_gain_list = vcat(mof_gain_list, mof_gain)
        if left_node !== nothing
            queue_compute_nodes!(queue, left_node)
            comp = vcat(comp, left_node.locate)
        end
        if right_node !== nothing
            queue_compute_nodes!(queue, right_node)
            comp = vcat(comp, right_node.locate)
        end
    end
    
    tree.root = root
    tree.comp = comp
    tree.depth = maximum(length.(comp))
    tree.leaves = count_leaves(tree.comp)
    tree.regions = tree.leaves
    tree.mof_gain = mof_gain_list
    tree.ne_type = ne_type
    tree.fm_type = fm_type
    println("time for creating one tree: ", time()-s_time)
    return tree
end

# Tree-level prediction funtion
function predict_single_tree(tree::Tree, X, Q)
    Y_hat = zeros(size(X)[2])
    for r in range(1, length(Y_hat), step = 1)
        node = tree.root
        while node.feature !== -1 && node.left_child !== nothing
            split_value = node.comp_value
            @views val = Q[node.feature, r]
            if val <= split_value
                node = node.left_child               
            else
                node = node.right_child 
            end          
        end
        if typeof(node.ne) <: GradEstimator
            Y_hat[r] = predict_grad(node.ne, X[2:end, r])[1]
        else
            Y_hat[r] = predict(node.ne, X[:, r])
        end
    end
    return Y_hat
end



# Pruning and Joining Helper Functions
function init_terminal_split(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 3) 
    end
    return term
end

function init_terminal_nodes(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 1) 
    end
    return term
end

# Group data Points
function data_groups(tree::Tree, Q, group_labels)
    groups = Array{Int64}(undef, 0)
    terminal_nodes = tree.comp[init_terminal_nodes(tree.comp)]
    
    for r in range(1, length(tree.root.data_idx), step = 1)
        location = ""
        node = tree.root
        while node.feature !== -1 && node.left_child !== nothing
            split_value = node.comp_value
            @views val = Q[node.feature, r]
            if val <= split_value
                node = node.left_child
                location = string(location, "0")               
            else
                node = node.right_child
                location = string(location, "1") 
            end          
        end
        push!(groups, group_labels[terminal_nodes .== location][1])
    end
    
    return groups
end

function find_prune(term_splits, mof_gain, fm_type)
    candidates = findall(x -> x == 1, term_splits)
    fm_obj = fm_type()
    fm_get_worst_mof!(fm_obj)  
    low_index = 0
    for x in candidates
        if fm_compare_prune(mof_gain[x], fm_obj)
            low_index = x
            fm_obj.mof = mof_gain[x]
        end
    end
    return low_index, fm_obj.mof
end

function get_data_indices_mof(tree::Tree, leaf_comp)
    data_idx_leaves = Vector{Vector{Int64}}()
    mof_leaves = Vector{typeof(tree.fm_type().mof)}()
    ne_list = []
    for leaf in leaf_comp
        node = tree.root
        j = 1
        while node.feature !== -1
            if leaf[j] == '0'
                node = node.left_child
                j = j + 1
            else
                node = node.right_child
                j = j + 1
            end
        end
        push!(data_idx_leaves, node.data_idx)
        push!(mof_leaves, node.fm.mof)
        push!(ne_list, node.ne)
    end
    return data_idx_leaves, mof_leaves, ne_list
end

function find_rj(pot_idx, mof_list, num_groups, groups, X, Y, ne_type, fm_type)
    # Running Objects
    fm_obj = fm_type()
    ne_obj = ne_type()
    # Best Objects
    fm_obj_opt = fm_type()
    fm_get_worst_mof!(fm_obj_opt)
    fm_obj_opt.mof_gain = fm_obj_opt.mof
    ne_obj_opt = ne_type()

    mgs = Int.(zeros(2))
    
    for i in 2:num_groups
        for j in 1:i
            if pot_idx[i, j] != []
                fit!(ne_obj, X[:, pot_idx[i, j]], Y[pot_idx[i, j]])
                Y_ij_hat  = predict(ne_obj, X[:, pot_idx[i, j]])
                fm_calc_mof!(fm_obj, Y[pot_idx[i, j]], Y_ij_hat)
                fm_obj.mof_gain = fm_obj.mof - unique(mof_list[groups .== i])[1] - unique(mof_list[groups .== j])[1]
                if fm_obj_opt.mof_gain > fm_obj.mof_gain
                    mgs = [j,i]
                    fm_obj_opt = deepcopy(fm_obj)
                    ne_obj_opt = deepcopy(ne_obj)
                end
            end
        end
    end
    
    mgs = sort(mgs)
    mg_1 = mgs[1]
    mg_2 = mgs[2]
    return mg_1, mg_2, fm_obj_opt, ne_obj_opt
end

# Merges Indicies from all leaves belonging to  a given grouping
function merge_indices(vec_of_indices)
    index_list = []

    for index in vec_of_indices
        index_list = vcat(index_list, index)
    end

    return index_list
end

# Updates the groups to reflect a new region join
function update_groups(groups, mg_1, mg_2)
    for i in 1:length(groups)
        if groups[i] == mg_2
            groups[i] = mg_1
        end
    end
    return groups
end

# Update Tree so new coefficients are reflected in the tree
function update_tree(tree::Tree, update_leaves, ne_merge)
    for leaf in update_leaves
        node = tree.root
        j = 1
        while node.feature !== -1
            if leaf[j] == '0'
                node = node.left_child
                j = j + 1
            else
                node = node.right_child
                j = j + 1
            end
        end
        node.ne = ne_merge
    end
    return tree
end

function find_terminal_node(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 1) 
    end
    return term
end

function update_tree_grad(tree::Tree, update_leaves, ne_merge)
    for leaf in update_leaves
        node = tree.root
        j = 1
        while node.feature !== -1
            if leaf[j] == '0'
                node = node.left_child
                j = j + 1
            else
                node = node.right_child
                j = j + 1
            end
        end
        node.ne.coeff = ne_merge.coeff
    end
    return tree
end

function convert_to_grad!(tree, X, Y)
    for loc in tree.comp
        node = tree.root
        for i in 1:length(loc)
            if loc[i] == '0'
                node = node.left_child
            else
                node = node.right_child
            end
        end
        fit_grad!(node.ne, X[:, node.data_idx], Y[node.data_idx])
    end

    leaf_comp = tree.comp[find_terminal_node(tree.comp)]
    for leaf in leaf_comp
        node = tree.root
        for i in 1:length(leaf)
            if leaf[i] == '0'
                node = node.left_child
            else
                node = node.right_child
            end
        end
        X[:, node.data_idx]  = X[:, node.data_idx] .- node.ne.X_demean
        Y[node.data_idx] = Y[node.data_idx] .- node.ne.Y_demean
    end
end
# Cross Validation Helper Functions
function n_fold_sample(nfolds, indices)
    indices = shuffle(indices)
    n = length(indices)
    fold_vec = Vector{Vector{Int64}}()
    obs_per_fold = convert(Int64, floor(n / nfolds))
    for i in 1 : nfolds
        if i * obs_per_fold + obs_per_fold > n
            push!(fold_vec, indices[(i -1) * obs_per_fold+1:end])
        else
            push!(fold_vec, indices[(i-1) * obs_per_fold + 1: (i-1) * obs_per_fold + obs_per_fold])
        end
    end
    return fold_vec
end

# Pruning and Pruning Based Cross Validation
function prune_full_tree(tree::Tree,Y)
    # Start timer
    s_time = time()

    # Keep tree input unedited
    tree = deepcopy(tree)

    # Keep list of trees and their mof gain from previous tree
    tree_list = Vector{Tree}()
    fm_obj = tree.fm_type()
    mof_gain_list = Vector{typeof(fm_obj.mof)}()

    # Full tree is chosen when no regularization
    push!(tree_list, deepcopy(tree))
    push!(mof_gain_list, fm_full_tree_reg_val(fm_obj))
    
    # Get sample size
    n = length(tree.root.data_idx)
    
    # Find all terminal splits
    term_splits = init_terminal_split(tree.comp)
   
    # Do until left with no splits
    while tree.root.left_child !== nothing
        # Get index of node that has lowest mof gain and then get its bin string
        prune_index, mof = find_prune(term_splits, tree.mof_gain, tree.fm_type)
        prune_bin = tree.comp[prune_index]
        
        node = tree.root
        # Navigate to correct split using binary string
        for x in prune_bin
            if x == '0'
                node = node.left_child
            elseif x == '1'
                node = node.right_child
            end
        end

        # Set children to nothing and feature to -1 so it is now a leaf
        node.left_child = nothing
        node.right_child = nothing
        node.feature = -1
        node.fm.mof_gain = fm_get_worst_mof!(fm_obj)
        node.comp_value = NaN
        
        # Change it to a leaf in term_splits\mof_gain lists (worst fit)
        term_splits[prune_index] = 0
        fm_get_worst_mof!(fm_obj)
        tree.mof_gain[prune_index] = fm_obj.mof

        # If sibling is a leaf, add parent to list of terminal splits
        if prune_bin !== ""
            if prune_bin[end] == '1'
                if node.parent.left_child.feature == -1
                    term_splits[findall(x -> x == prune_bin[1:end-1], tree.comp)] = [1]
                end
            elseif prune_bin[end] == '0'
                if node.parent.right_child.feature == -1
                    term_splits[findall(x -> x == prune_bin[1:end-1], tree.comp)] = [1]
                end
            end
        end

        # Remove children of new leaf from all lists
        children = [string(prune_bin,'0'), string(prune_bin,'1')]
        for child in children
            child_index = findall(x -> x == child, tree.comp)[1]
            deleteat!(term_splits, child_index)
            deleteat!(tree.mof_gain, child_index)
            deleteat!(tree.comp, child_index)
        end
        tree.depth = maximum(length.(tree.comp))
        tree.leaves = count_leaves(tree.comp)
        tree.regions = tree.leaves

        # Add tree to list
        push!(mof_gain_list, mof)
        push!(tree_list, deepcopy(tree))
        
    end
    

    # Intiialze indices that point to which tree is optimal for a given regularization parameter
    tree_final_index = Vector{Int64}()
    # Get normalized mof from the gains 
    normalized_mof = cumsum(mof_gain_list)
    # Make grid of regularization values
    regularize_final = [x for x in range(.5, 3, length = 20)]
    # This line adds the specific regularization values to the grid as well 
    # regularize_final = sort(unique(vcat(regularize_final, mof_gain))) 

    # Create penaliztaion term as matrix
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(tree_final_index, findmin(normalized_mof .+ penalized_term[i,:])[2])
    end
    println("time for pruning one tree: ", time()-s_time)
    return tree_list, tree_final_index, regularize_final./n
end

function prune_val(tree::Tree, regularize_final, Y)
    # Start timer
    s_time = time()

    # Keep tree input unedited
    tree = deepcopy(tree)

    # Keep list of trees and their mof gain from previous trees
    tree_list = Vector{Tree}()
    fm_obj = tree.fm_type()
    mof_gain = Vector{typeof(fm_obj.mof)}()

    # Full tree is chosen when no regularization
    push!(tree_list, deepcopy(tree))
    push!(mof_gain, fm_full_tree_reg_val(fm_obj))
    
    # Get sample size
    n = length(tree.root.data_idx)
    
    # Find all terminal splits
    term_splits = init_terminal_split(tree.comp)
    
    # Do until left with no splits
    while tree.root.left_child !== nothing
        # Get index of node that has lowest sse gain and then get its bin string
        prune_index, mof = find_prune(term_splits, tree.mof_gain, tree.fm_type)
        prune_bin = tree.comp[prune_index]
        node = tree.root
        # Navigate to correct split using binary string
        for x in prune_bin
            if x == '0'
                node = node.left_child
            elseif x == '1'
                node = node.right_child
            end
        end

        # Set children to nothing and feature to -1 so it is now a leaf
        node.left_child = nothing
        node.right_child = nothing
        node.feature = -1
        node.fm.mof_gain = fm_get_worst_mof!(fm_obj)
        node.comp_value = NaN
        
        # Change it to a leaf in term_splits\mof_gain lists (worst fit)
        term_splits[prune_index] = 0
        fm_get_worst_mof!(fm_obj)
        tree.mof_gain[prune_index] = fm_obj.mof

        # If sibling is a leaf, add parent to list of terminal splits
        if prune_bin !== ""
            if prune_bin[end] == '1'
                if node.parent.left_child.feature == -1
                    term_splits[findall(x -> x == prune_bin[1:end-1], tree.comp)] = [1]
                end
            elseif prune_bin[end] == '0'
                if node.parent.right_child.feature == -1
                    term_splits[findall(x -> x == prune_bin[1:end-1], tree.comp)] = [1]
                end
            end
        end
        
        # Remove children of new leaf from all lists
        children = [string(prune_bin,'0'), string(prune_bin,'1')]
        for child in children
            child_index = findall(x -> x == child, tree.comp)[1]
            deleteat!(term_splits, child_index)
            deleteat!(tree.mof_gain, child_index)
            deleteat!(tree.comp, child_index)
        end
        tree.depth = maximum(length.(tree.comp))
        tree.leaves = count_leaves(tree.comp)
        tree.regions = tree.leaves

        # Add tree to list
        push!(mof_gain, mof)
        push!(tree_list, deepcopy(tree))
        
    end
    

    # Get normalized mof from the gains 
    normalized_mof = cumsum(mof_gain)
    # Rescale regularizaiton parameters
    regularize_final = regularize_final .* n
    # Initialize list of trees
    final_tree_list = Vector{Tree}()

    # Create penalization term as matrix
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(final_tree_list, tree_list[findmin(normalized_mof .+ penalized_term[i,:])[2]])
    end
    println("time for pruning one CV tree: ", time()-s_time)
    return final_tree_list
end

function nfold_cross_val_prune(X, Y, Q, regularize_final, nfolds, minobs, ne_type, fm_type)
    if ndims(Q) == 1
        Q = reshape(Q, 1, length(Q))
    end
    indices = [x for x in 1:length(Y)]
    # Create partition of samples
    fold_vec = n_fold_sample(nfolds, indices)
    total_nfeatures = size(Q,1)
    feature_idx = [x for x in 1:total_nfeatures]
    # Keep Track of SSE for each alpha\validation set
    mof_tally = zeros(typeof(fm_type().mof), nfolds, length(regularize_final))
    # For each validation set
    for i in 1:nfolds
        println("Starting fold ", i)
        # Make training set out of other folds
        train_ind = deepcopy(indices)
        deleteat!(train_ind, sort(fold_vec[i]))
        # Train the tree on the training data
        tree = grow_tree(X[:,train_ind], Y[train_ind], Q[:, train_ind], feature_idx, minobs, ne_type, fm_type)
        
        # Prune tree based on full tree alpha levels
        val_trees = prune_val(tree, regularize_final, Y)

        predict_val(x) = predict_single_tree(x, X[:, fold_vec[i]], Q[:,fold_vec[i]])
        Y_hat_list = predict_val.(val_trees)
        for j in 1:length(regularize_final)
            fm_obj = fm_type()
            fm_calc_mof!(fm_obj, Y[fold_vec[i]], Y_hat_list[j])
            mof_tally[i,j] = fm_obj.mof
        end
    end
    mof_totals= sum(mof_tally,dims = 1)[1,:]
    return findall(x -> x == minimum(mof_totals), mof_totals)
end

function rj_full_tree(tree::Tree, X, Y)
    # Start timer
    s_time = time()
    
    # Keep tree input unedited
    tree = deepcopy(tree)

    # Keep list of trees and their associated mof gains, estimators, and groupingss
    tree_list = Vector{Tree}()
    fm_obj = tree.fm_type()
    mof_gain_list = Vector{typeof(fm_obj.mof)}()
    ne_list = []
    group_list = Vector{Vector{Int64}}()
    
    # Get sample size
    n = length(tree.root.data_idx)
    d,~ = size(X)
    # gind terminal nodes (leaves)
    term_node_bool = find_terminal_node(tree.comp)
    T = sum(term_node_bool)
    leaf_comp = tree.comp[term_node_bool]

    # If this is a gradient estimation, remove constant term
    if tree.ne_type <: GradEstimator

        X = X[2:end, :]
        convert_to_grad!(tree, X, Y)
    end

    # Initialize variables used to encode which regions are joined and to keep track of mof, nodes estimators, and data indices
    groups = [x for x in 1:T]
    indices, mof_leaves, ne_group = get_data_indices_mof(tree, leaf_comp)
    num_groups = length(unique(groups))

    

    

    # Push first element of each list (original tree, alpha of 0, original θs, and all unique groups)
    push!(tree_list, deepcopy(tree))
    push!(mof_gain_list, deepcopy(fm_full_tree_reg_val(fm_obj)))
    push!(ne_list, deepcopy(ne_group))
    push!(group_list, deepcopy(groups))
    
    # Begin loop which ends when all regions are grouped together
    
    while groups != Int.(ones(T))
        # Construct the data indices for each potential join
        pot_idx = fill(Int[], num_groups, num_groups)
        u_groups = sort(unique(groups))
        for i in 2:num_groups
            for j in 1:i
                if u_groups[i] != u_groups[j]
                    index_i = merge_indices(indices[groups .== u_groups[i]])
                    index_j = merge_indices(indices[groups .== u_groups[j]])
                    pot_idx[i, j] = sort(unique(vcat(index_i, index_j)))
                end
            end
        end
        
        
        # Find best join and output the two regions that are being joined
        mg_1, mg_2, fm_update, ne_merge = find_rj(pot_idx, mof_leaves, num_groups, groups, X, Y, tree.ne_type, tree.fm_type)
        
        # Update Groups vector,  number of groups, sse, and θgroup
        groups = update_groups(groups, mg_1, mg_2)
        num_groups = length(unique(groups))
        mof_leaves[groups .== mg_1] .= fm_update.mof
        updated_indices = findall(groups .== mg_1)
        update_leaves = leaf_comp[groups .== mg_1]
        # If gradient estimator, only updated the coeffcient. Otherwise just change the estimator.
        if tree.ne_type <: GradEstimator
            for x in updated_indices
                ne_group[x].coeff = ne_merge.coeff
            end
            tree = deepcopy((update_tree_grad(tree, update_leaves, ne_merge)))
        else
            for x in updated_indices
                ne_group[x] = ne_merge
            end
            tree = deepcopy((update_tree(tree, update_leaves, ne_merge)))
        end

        # Update group numberings so any single group number is not larger than the number of groups there are
        if sum(groups .> num_groups) != 0
            over_groups = sort(unique(groups[groups .> num_groups]))
            missing_groups = [x for x in 1:num_groups]
            missing_groups = sort(setdiff(missing_groups, groups))
            for i in 1:length(groups)
                if groups[i] > num_groups
                    groups[i] = missing_groups[over_groups .== groups[i]][1]
                end            
            end
        end
        
        # Update Tree Parameters to reflect grouping and push that tree to treelist 
        tree.regions = num_groups
        push!(tree_list, deepcopy(tree))
        push!(mof_gain_list, deepcopy(fm_update.mof_gain))
        push!(ne_list, deepcopy(ne_group))
        push!(group_list, deepcopy(groups))
        
        
    end
    
    # Now make richer list of alphas and do not assume that each merge increases sse by more even though it may be generally likely.
    treefinalindex = Vector{Int64}()
    normalized_mof = cumsum(mof_gain_list)
    ## Standard
    #regularize_final = [x for x in range(0, stop = 1.1*(maximum(mof_gain_list)), length = 2 * length(tree_list))]
    #regularize_final = sort(unique(vcat(regularize_final, mof_gain_list)))
    #penalized_term = [x * y.regions for x = regularize_final, y in tree_list]
    #regularize_final = regularize_final./n
    ## BIC 
    regularize_final = [x for x in range(.5, 3, length = 20)]
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(treefinalindex, findmin(normalized_mof .+ penalized_term[i,:])[2])
    end
    println("time for joining one tree: ", time()-s_time)
    #return tree_list, group_list, ne_list, treefinalindex, regularize_final, indices
    return tree_list, group_list, ne_list, treefinalindex, regularize_final, indices
end


# Validation region joining, taking regularization paramaters as given
function val_rj(tree, regularize_final, X, Y)
    # Start timer
    s_time = time()

    # Keep tree input unedited
    tree = deepcopy(tree)

    # Keep list of trees, their mof gain from previous tree, their estimators, and the groupings
    tree_list = Vector{Tree}()
    fm_obj = tree.fm_type()
    mof_gain_list = Vector{typeof(fm_obj.mof)}()
    ne_list = [] 
    group_list = Vector{Vector{Int64}}()
    
    # Sample size of tree
    n = length(tree.root.data_idx)
    
    # Get terminal node information
    term_node_bool = find_terminal_node(tree.comp)
    T = sum(term_node_bool)
    leaf_comp = tree.comp[term_node_bool]

    # If this is a gradient estimation, remove constant term
    if tree.ne_type <: GradEstimator
        X = X[2:end, :]
        convert_to_grad!(tree, X, Y)
    end

    # Initialize variables used to encode which regions are joined and to keep track of sse, θs, and data indices
    groups = [x for x in 1:T]
    indices, mof_leaves, ne_group = get_data_indices_mof(tree, leaf_comp)
    
    num_groups = length(unique(groups))
    # Push first element of each list (original tree, alpha of 0, original θs, and all unique groups)
    push!(tree_list, deepcopy(tree))
    push!(mof_gain_list, deepcopy(fm_full_tree_reg_val(fm_obj)))
    push!(ne_list, deepcopy(ne_group))
    push!(group_list, deepcopy(groups))
    
    # Begin loop which ends when all regions are grouped together
    
    while groups != Int.(ones(T))
        # Construct the data indices for each potential join
        pot_idx = fill(Int[], num_groups, num_groups)
        u_groups = sort(unique(groups))
        for i in 2:num_groups
            for j in 1:i
                if u_groups[i] != u_groups[j]
                    index_i = merge_indices(indices[groups .== u_groups[i]])
                    index_j = merge_indices(indices[groups .== u_groups[j]])
                    pot_idx[i, j] = sort(unique(vcat(index_i, index_j)))
                end
            end
        end
        
        
        # Find best join and output the two regions that are being joined
        mg_1, mg_2, fm_update, ne_merge = find_rj(pot_idx, mof_leaves, num_groups, groups, X, Y, tree.ne_type, tree.fm_type)
        
        # Update Groups vector,  number of groups, sse, and θgroup
        groups = update_groups(groups, mg_1, mg_2)
        num_groups = length(unique(groups))
        mof_leaves[groups .== mg_1] .= fm_update.mof
        updated_indices = findall(groups .== mg_1)
        update_leaves = leaf_comp[groups .== mg_1]
        # If gradient estimator, only updated the coeffcient. Otherwise just change the estimator.
        if tree.ne_type <: GradEstimator
            for x in updated_indices
                ne_group[x].coeff = ne_merge.coeff
            end
            tree = deepcopy((update_tree_grad(tree, update_leaves, ne_merge)))
        else
            for x in updated_indices
                ne_group[x] = ne_merge
            end
            tree = deepcopy((update_tree(tree, update_leaves, ne_merge)))
        end
        
        

        # Update group numberings so any single group number is not larger than the number of groups there are
        if sum(groups .> num_groups) != 0
            over_groups = sort(unique(groups[groups .> num_groups]))
            missing_groups = [x for x in 1:num_groups]
            missing_groups = sort(setdiff(missing_groups, groups))
            for i in 1:length(groups)
                if groups[i] > num_groups
                    groups[i] = missing_groups[over_groups .== groups[i]][1]
                end            
            end
        end

        
        tree.regions = num_groups
        push!(tree_list, deepcopy(tree))
        push!(mof_gain_list, deepcopy(fm_update.mof_gain))
        push!(ne_list, deepcopy(ne_group))
        push!(group_list, deepcopy(groups))
        
        
    end
    
    
    final_tree_list = Vector{Tree}()
    normalized_mof = cumsum(mof_gain_list)
    
    ## Standard
    #regularize_final = n * regularize_final 
    #penalized_term = [x * y.regions  for x = regularize_final, y in tree_list]
    ## BIC
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(final_tree_list, tree_list[findmin(normalized_mof .+ penalized_term[i,:])[2]])
    end
    println("time for joining CV tree: ", time()-s_time)
    return final_tree_list
end

# Cross Validation for region joining 
function nfold_cross_val_rj(X, Y, Q, regularize_final, nfolds, minobs, ne_type, fm_type)
   
    if ndims(Q) == 1
        Q = reshape(Q, 1, length(Q))
    end
    indices = [x for x in 1:length(Y)]
    # Create partition of samples
    fold_vec = n_fold_sample(nfolds, indices)
    total_nfeatures = size(Q,1)
    feature_idx = [x for x in 1:total_nfeatures]
    # Keep Track of MOF for each alpha\validation set
    mof_tally = zeros(typeof(fm_type().mof), nfolds, length(regularize_final))
    # For each validation set
    for i in 1:nfolds
        println("Starting fold ", i)
        # Make training set out of other folds
        train_ind = deepcopy(indices)
        deleteat!(train_ind, sort(fold_vec[i]))
        # Train the tree on the training data
        tree = grow_tree(X[:,train_ind], Y[train_ind], Q[:, train_ind], feature_idx, minobs, ne_type, fm_type)
        # Prune tree based on full tree alpha levels
        val_trees = val_rj(tree, regularize_final, X[:, train_ind], Y[train_ind])
        nval = length(Y[train_ind])
        predict_val(x) = predict_single_tree(x, X[:,fold_vec[i]], Q[:,fold_vec[i]])
        Y_hat_list = predict_val.(val_trees)
        for j in 1:length(regularize_final)
            fm_obj = fm_type()
            fm_calc_mof!(fm_obj, Y[fold_vec[i]], Y_hat_list[j])
            mof_tally[i,j] = fm_obj.mof
        end
    end
    mof_totals= sum(mof_tally,dims = 1)[1,:]
    return findall(x -> x == minimum(mof_totals), mof_totals)
end

function BIC(joined_trees, Y, X, Q_tilde)
    n = length(Y)
    counter = 1
    d,~ = size(X)
    BICs = zeros(length(joined_trees))
    for tree in joined_trees
        mhat = tree.regions
        Yhat = predict_single_tree(tree, X, Q_tilde)
        u =(Y - Yhat)
        sse = sum((u).^2)
        var_y = var(Y)
        BICs[counter] = sse + d * log(n) * mhat * var_y
        counter += 1
    end
    return argmin(BICs)
end


# Interpretation Algorithims
function get_tree_hyperrectangles(tree)
    term_node_bool = find_terminal_node(tree.comp)
    leaf_comp = tree.comp[term_node_bool]
    # First Construct Q
    Q = []
    for leaf in leaf_comp
        node = tree.root
        j = 1
        while node.feature !== -1
            push!(Q,node.feature)
            if leaf[j] == '0'
                node = node.left_child
                j = j + 1
            else
                node = node.right_child
                j = j + 1
            end
        end
    end
    Q = sort(unique(Q))
    γu = []
    γl = []
    param = []
    
    for leaf in leaf_comp
        node = tree.root
        γuk = Inf * ones(length(Q))
        γlk = -Inf * ones(length(Q))
        j = 1
        while node.feature !== -1
            qindex = findall(x -> x == node.feature, Q)[1]
            if leaf[j] == '0'
                γuk[qindex] = node.comp_value
                node = node.left_child
                j = j + 1
            else
                γlk[qindex] = node.comp_value
                node = node.right_child
                j = j + 1
            end
        end
        push!(param, node.ne)
        push!(γu,γuk)
        push!(γl,γlk)
    end


    return Q, param, γl, γu
end

function compare_thresholds(γlj, γuj, γlk, γuk, dim_thresh)
    if sum(γlj .== γlk) == dim_thresh && sum(γuj .== γuk) == dim_thresh
        if sum(γuj .== γlk) == 1
            return 1
        end
        if sum(γlj .== γuk) == 1
            return -1
        end
    end
    return 0
end

function get_largest_hyperrectangles(groups, Q, γl, γu)
    dim_thresh = length(Q) - 1
    γl_final = Array{Array{Array{Float64,1},1},1}()
    γu_final = Array{Array{Array{Float64,1},1},1}()
    for i in 1:length(unique(groups))
        γltemp = γl[(groups .== i)]
        γutemp = γu[(groups .== i)]
        
        @label top
        
        for j in range(1, length(γltemp), step = 1)
            for k in 1:j-1
                mergable = compare_thresholds(γltemp[j], γutemp[j], γltemp[k], γutemp[k], dim_thresh)
                if abs(mergable) == true
                    # Modify one region's borders
                    if mergable == 1
                        index = findall(x -> x == 1, γutemp[j] .== γltemp[k])
                        γutemp[j][index] = γutemp[k][index]
                    elseif mergable == -1
                        index = findall(x -> x == 1, γltemp[j] .== γutemp[k])
                        γltemp[j][index] = γltemp[k][index]
                    end

                    # Delete one region's borders
                    deleteat!(γltemp, k)
                    deleteat!(γutemp, k)
                    @goto top
                end
            end
        end

       push!(γl_final, γltemp)
       push!(γu_final, γutemp) 
    end

    # Clean Noisy Dimensions
    qindex_delete = Array{Int64,1}()
    for i in 1:length(Q)
        non_inf_i = 0
        # Check Lower bounds for all neg infinity
        for group_bounds ∈ γl_final
            for group_bound ∈ group_bounds
                if group_bound[i] != -Inf
                    non_inf_i = 1
                end
            end
        end

        # Check Upper bounds for all infinity
        for group_bounds ∈ γu_final
            for group_bound ∈ group_bounds
                if group_bound[i] != Inf
                    non_inf_i = 1
                end
            end
        end
        if non_inf_i == 0
            push!(qindex_delete, i)
        end
    end

    # Delete Noisy Dimension Threshold Information
    if size(qindex_delete)[1] != 0
        deleteat!(Q, qindex_delete)
        # Delete Lower bounds
        for group_bounds ∈ γl_final
            for rec_bounds in group_bounds
                deleteat!(rec_bounds, qindex_delete)
            end
        end

        # Delete Upper bounds
        for group_bounds ∈ γu_final
            for rec_bounds in group_bounds
                deleteat!(rec_bounds, qindex_delete)
            end
        end
    end

    return Q, γl_final, γu_final
end

function get_thresholds_from_borders(γl, γu)
    num_groups = length(γl)
    Γ_above = Vector{Int64}()
    Γ_below = Vector{Int64}()
    thresh_vars = Vector{Int64}()
    thresh_vals = Vector{Float64}()
    low_endpoints = Vector{Vector{Vector{Float64}}}()
    high_endpoints = Vector{Vector{Vector{Float64}}}()
    
    for i ∈ 2:num_groups
        for j ∈ 1:(i-1)
            # j above i
            for k ∈ 1:length(γu[i])
                for l ∈ 1:length(γl[j])
                    # Check to see if they share only one element
                    if sum((γu[i][k] .== γl[j][l])) == 1
                        # Get threhsold variable and value
                        thresh_var = findmax((γu[i][k] .== γl[j][l]))[2]
                        thresh_val = γu[i][k][thresh_var]
                        if sum((γu[i][k] .>= γl[j][l])) == length(γl[j][l]) && sum((γu[j][l] .>= γl[i][k])) == length(γu[j][l]) 
                            #println(thresh_var, " ",thresh_val, " ", j, " above ", i)
                            #println(γl[i][k], " ", γu[i][k], " ", γl[j][l], " ", γu[j][l])
                            # Find endpoints of threshold
                            low_endpoint = maximum(hcat(γl[j][l], γl[i][k]), dims = 2)[:, 1]
                            high_endpoint = minimum(hcat(γu[j][l], γu[i][k]), dims = 2)[:, 1]
                            # Check if threhsold already accounted for
                            if length(Γ_above) == 0
                                thresh_exists = 0
                            else
                                check_groups = (Γ_above .== j) .* (Γ_below .== i)
                                check_thresh = (thresh_vars .== thresh_var) .* (thresh_vals .== thresh_val)
                                check_location = (check_groups .* check_thresh)
                                thresh_exists = sum(check_location)
                                
                            end
                            # Update lists
                            if thresh_exists == 1
                                #println(low_endpoints[check_location], [low_endpoint])
                                
                                push!(low_endpoints[check_location][1,1], low_endpoint)
                                push!(high_endpoints[check_location][1,1], high_endpoint)
                            else
                                push!(Γ_above, j)
                                push!(Γ_below, i)
                                push!(thresh_vars, thresh_var)
                                push!(thresh_vals, thresh_val)
                                push!(low_endpoints, [low_endpoint])
                                push!(high_endpoints, [high_endpoint])
                            end
                        end
                    end
                end
            end
    
            # j below i
            for k ∈ 1:length(γu[i])
                for l ∈ 1:length(γl[j])
                    # Check to see if they share only one element
                    if sum((γl[i][k] .== γu[j][l])) == 1
                        # Get threhsold variable and value
                        thresh_var = findmax((γl[i][k] .== γu[j][l]))[2]
                        thresh_val = γl[i][k][thresh_var]
                        if sum((γu[i][k] .>= γl[j][l])) == length(γl[j][l]) && sum((γu[j][l] .>= γl[i][k])) == length(γu[j][l])
                            #println(thresh_var, " ",thresh_val, " ", j, " below ", i)
                            #println(γl[j][l], " ", γu[j][l], " ", γl[i][k], " ", γu[i][k])
                            # Find endpoints of threshold
                            low_endpoint = maximum(hcat(γl[j][l], γl[i][k]), dims = 2)[:, 1]
                            high_endpoint = minimum(hcat(γu[j][l], γu[i][k]), dims = 2)[:, 1]
                            # Check if threhsold already accounted for
                            if length(Γ_above) == 0
                                thresh_exists = 0
                            else
                                check_groups = (Γ_above .== i) .* (Γ_below .== j)
                                check_thresh = (thresh_vars .== thresh_var) .* (thresh_vals .== thresh_val)
                                check_location = (check_groups .* check_thresh)
                                thresh_exists = sum(check_location)
                            end
                            # Update lists
                            if thresh_exists == 1
                                #println(low_endpoints[check_location], [low_endpoint])
                                push!(low_endpoints[check_location][1,1], low_endpoint)
                                push!(high_endpoints[check_location][1,1], high_endpoint)
                            else
                                push!(Γ_above, i)
                                push!(Γ_below, j)
                                push!(thresh_vars, thresh_var)
                                push!(thresh_vals, thresh_val)
                                push!(low_endpoints, [low_endpoint])
                                push!(high_endpoints, [high_endpoint])
                            end
                        end
                    end
                end
            end
        end
    end

    thresh_objects = Vector{Threshold}()
    for t in 1:length(Γ_below)
        for j in 2:length(low_endpoints[t])
            for k in 1:j
                if sum(low_endpoints[t][j] .== high_endpoints[t][k]) == 2 && sum(high_endpoints[t][j] .== high_endpoints[t][k]) == (length(high_endpoints[t][k]) - 1) && sum(low_endpoints[t][j] .== low_endpoints[t][k]) == (length(low_endpoints[t][k]) - 1) 
                    high_endpoints[t][k] = high_endpoints[t][j]
                    low_endpoints[t][j] = low_endpoints[t][k]
                elseif sum(high_endpoints[t][j] .== low_endpoints[t][k]) == 2 && sum(high_endpoints[t][j] .== high_endpoints[t][k]) == (length(high_endpoints[t][k]) - 1) && sum(low_endpoints[t][j] .== low_endpoints[t][k]) == (length(low_endpoints[t][k]) - 1) 
                    high_endpoints[t][j] = high_endpoints[t][k]
                    low_endpoints[t][k] = low_endpoints[t][j]
                end
            end

        end
        endpoints_temp = unique(hcat(low_endpoints[t], high_endpoints[t]), dims = 1)
        low_endpoints[t] = endpoints_temp[:,1]
        high_endpoints[t] = endpoints_temp[:,2]
        push!(thresh_objects, Threshold(Γ_above[t], Γ_below[t], thresh_vars[t], thresh_vals[t], low_endpoints[t], high_endpoints[t]))
    end
    return thresh_objects
end

function get_thresholds(tree, groups)
    Qindexs, ne, γls, γus = get_tree_hyperrectangles(tree)
    Qindex, γl, γu = get_largest_hyperrectangles(groups, Qindexs, γls, γus)
    return get_thresholds_from_borders(γl, γu), Qindex
end
end



