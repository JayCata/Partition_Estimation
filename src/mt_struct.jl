#------------------------------------------------------#
### Packages ###
#------------------------------------------------------#

using Statistics
using SparseArrays
using LinearAlgebra
using Random
using Plots
using FixedEffects: FixedEffect, solve_residuals!
using StatsBase: Weights
import Base.show


#------------------------------------------------------#
### Node Estimator Types ###
#------------------------------------------------------#

# Abstract Types for Node Level Estimators
abstract type  NodeEstimator
end

# Abstract Type for Gradient Estimator
abstract type GradEstimator <: NodeEstimator
end

# Concrete OLS Node Estimator
mutable struct ols <: NodeEstimator
    coeff       :: Union{Float64, Vector{Float64}}
    se          :: Union{Float64, Vector{Float64}}
    ols() = new()
end

# Weighted OLS
mutable struct w_ols <: NodeEstimator
    coeff       :: Union{Float64, Vector{Float64}}
    se          :: Union{Float64, Vector{Float64}}
    w_ols() = new()
end

# FE OLS
mutable struct fe_ols <: NodeEstimator
    coeff       :: Union{Float64, Vector{Float64}}
    se          :: Union{Float64, Vector{Float64}}
    fe_ols() = new()
end

# Concrete OLS Grad Estimator
mutable struct ols_grad <: GradEstimator
    proj_coeff  :: Union{Float64, Vector{Float64}}
    coeff       :: Union{Float64, Vector{Float64}}
    X_demean    :: Union{Float64, Vector{Float64}}
    se          :: Union{Float64, Vector{Float64}}
    ols_grad() = new()
end


#------------------------------------------------------#
### NodeEstimator Methods ###
#------------------------------------------------------#

## OLS Node Estimator Methods ## 
# Fit OLS
function fit!(ne::ols, X::AbstractArray, Y::AbstractArray)
    if size(X, 1) != 1
        ne.coeff = inv(X * X') * (X * Y)[:,1]
    else
        ne.coeff = inv(X * X') * (X * Y)
    end
end

function fit_se!(ne::ols, X::AbstractArray, Y::AbstractArray)
    
    ne.se = sqrt.(diag(inv(X * X') * (sum((Y - X' * ne.coeff) .^ 2) * (1 / (length(Y) - size(X,1))))))
end

# Predict with OLS
function predict(ne::ols, X::AbstractArray)
    return X' * ne.coeff
end

## Weighted OLS Node Estimator Methods ## 
# Fit weighted OLS
function fit!(ne::w_ols, X::AbstractArray, Y::AbstractArray)
    W = Diagonal(X[end, :]/sum(X[end, :]))
    ne.coeff = (inv(X[1:end-1,:] * W * X[1:end-1,:]') * (X[1:end-1,:] * W * Y))[:,1]
end

function fit_se!(ne::w_ols, X::AbstractArray, Y::AbstractArray)
    ne.se = sqrt.(diag(inv(X[1:end-1,:] * X[1:end-1,:]') * (sum((Y - X[1:end-1,:]' * ne.coeff) .^ 2) * (1 / (length(Y) - size(X[1:end-1,:],1))))))
end

# Predict with weighted OLS
function predict(ne::w_ols, X::AbstractArray)
    return X[1:end-1,:]' * ne.coeff
end

## Fixed Effects OLS Node Estimator Methods ## 
# Fit fixed effects OLS
function fit!(ne::fe_ols, X::AbstractArray, Y::AbstractArray)
    YX, _, _ = solve_residuals!([deepcopy(Y) deepcopy(X[1:end-2, :]')], [FixedEffect(X[end-1, :])], Weights(X[end, :]))
    W = Diagonal(X[end, :]/sum(X[end, :]))
    ne.coeff = (inv(YX[:,2:end]' * W * YX[:,2:end]) * (YX[:, 2:end]' * W * YX[:,1]))[:, 1]
end

function fit_se!(ne::fe_ols, X::AbstractArray, Y::AbstractArray)
    # on hold
end

# Predict with weighted OLS
function predict(ne::fe_ols, X::AbstractArray)
    YX, _, _ = solve_residuals!(deepcopy(X[1:end-2, :])', [FixedEffect(X[end-1, :])], Weights(X[end, :]))
    return YX * ne.coeff
end


## OLS Grad Node Estimator Methods ##
# Fit OLS Grad
function fit!(ne::ols_grad, X::AbstractArray, Y::AbstractArray)
    ne.coeff = inv(X * X') * (X * Y)[:,1]
end

function fit_se!(ne::ols_grad, X::AbstractArray, Y::AbstractArray)
    # on hold
end

# Fit OLS Grad Demean
function fit_grad!(ne::ols_grad, X::AbstractArray, Y::AbstractArray, group_vars, proj_vars)
    ne.X_demean = mean(X[group_vars, :], dims = 2)[:, 1]

    # Do not demean intercept
    if 1 in group_vars
        ne.X_demean = zeros(length(group_vars))
    end
    ne.proj_coeff = ne.coeff[proj_vars]
    ne.coeff = inv((X[group_vars, :]' .- ne.X_demean')' * (X[group_vars, :]' .- ne.X_demean')) * ((X[group_vars, :]' .- ne.X_demean')' * (Y.- X[proj_vars, :]' * ne.proj_coeff))
    
end

# Predict OLS Grad
function predict(ne::ols_grad, X::AbstractArray)
    return X' * ne.coeff
end

# Predict OLS Grad Demean
function predict_grad( ne::ols_grad, X::AbstractArray, group_vars, proj_vars)
    return (X[group_vars,:]'.- ne.X_demean') * ne.coeff .+ (X[proj_vars, :]' * ne.proj_coeff)
end


#------------------------------------------------------#
### Fit Measure Types ###
#------------------------------------------------------#

# Asbtract Type for Fit Measure
abstract type FitMeasure
end

# Concrete SSE Fit Measure
mutable struct sse <: FitMeasure
    mof :: Float64
    mof_gain :: Union{Nothing, Float64}
    sse() = new()
end

# Concrete Peanlized SSE Fit Measure for parallel Trends
mutable struct pt_sse <: FitMeasure
    mof :: Float64
    mof_gain :: Union{Nothing, Float64}
    pt_sse() = new()     
end

#------------------------------------------------------#
### Fit Measure Methods ###
#------------------------------------------------------#
## Standard SSE Methods
# SSE Calculate Measure of Fit
function fm_calc_mof!(fm::sse, Y::AbstractArray, Y_hat::AbstractArray, X::AbstractArray,  n, ne)
    fm.mof = sum((Y - Y_hat).^2)
end

# SSE Obtain Worst Meaure of Fit
function fm_get_worst_mof!(fm::sse)
    fm.mof = Inf64
end

# SSE Compare During Tree Growing
function fm_compare_grow(l_fm::sse, r_fm::sse, best_l_fm::sse, best_r_fm::sse)
    return l_fm.mof + r_fm.mof < best_l_fm.mof + best_r_fm.mof
end

# SSE Compare During Tree Pruning
function fm_compare_prune(new_val, fm_obj::sse)
    if new_val < fm_obj.mof
        return true
    end
    return false
end

function stop_early(node, split_obj, X, Y, fm::sse)
    
    return (node.fm.mof - split_obj.left_fm.mof - split_obj.right_fm.mof) > (var(Y[node.data_idx]) * sqrt(length(Y[node.data_idx])))
end


# Normalized "Best" Measure of Fit
function fm_full_tree_reg_val(fm_obj::sse)
    return 0.0
end

## pt_sse calculate Measure of Fit
function fm_calc_mof!(fm::pt_sse, Y::AbstractArray, Y_hat::AbstractArray, X::AbstractArray, n, ne)
    int_coeffs = ne.coeff[Int(length(ne.coeff)/2):end]
    if typeof(ne) <: fe_ols
        Ytemp, _, _ =  solve_residuals!(deepcopy(Y), [FixedEffect(X[end-1, :])], Weights(X[end, :]))
        fm.mof = sum((Ytemp - Y_hat).^2) + ((sum(int_coeffs .^ 2)/length(int_coeffs) + sum(diff(int_coeffs) .^ 2)/length(diff(int_coeffs)) ))* (log(n)) * var(Ytemp)
    else
        fm.mof = sum((Y - Y_hat).^2) + ((sum(int_coeffs .^ 2)/length(int_coeffs) + sum(diff(int_coeffs).^ 2)/length(diff(int_coeffs)) )* (log(n))) 
    end
end

# SSE Obtain Worst Meaure of Fit
function fm_get_worst_mof!(fm::pt_sse)
    fm.mof = Inf64
end

# SSE Compare During Tree Growing
function fm_compare_grow(l_fm::pt_sse, r_fm::pt_sse, best_l_fm::pt_sse, best_r_fm::pt_sse)
    return l_fm.mof + r_fm.mof < best_l_fm.mof + best_r_fm.mof
end

# SSE Compare During Tree Pruning
function fm_compare_prune(new_val, fm_obj::pt_sse)
    if new_val < fm_obj.mof
        return true
    end
    return false
end

# Normalized "Best" Measure of Fit
function fm_full_tree_reg_val(fm_obj::pt_sse)
    return 0.0
end

function stop_early(node, split_obj, X, Y, fm::pt_sse)

    return ( sum((Y[node.data_idx] .- predict(node.ne, X[:, node.data_idx])) .^ 2 )- sum((Y[split_obj.left_idx] .- predict(split_obj.left_ne, X[:, split_obj.left_idx])) .^ 2)  - sum((Y[split_obj.right_idx] .- predict(split_obj.right_ne, X[:, split_obj.right_idx])) .^ 2 )) > (var(Y[node.data_idx]) * log(length(Y[node.data_idx])))
end

#------------------------------------------------------#
#------------------------------------------------------#

### Node and Tree Structures ###
# Split Object Structure
mutable struct SplitObj
    feature     :: Int64
    min         :: Float64
    max         :: Float64
    split_val   :: Float64
    left_idx    :: Vector{Int64}
    hleft_idx   :: Vector{Int64}
    right_idx   :: Vector{Int64}
    hright_idx  :: Vector{Int64}
    left_ne     :: NodeEstimator
    right_ne    :: NodeEstimator
    left_fm     :: FitMeasure
    right_fm    :: FitMeasure
    SplitObj() = new() 
end

# Node Structure
mutable struct Node
    feature     :: Int64
    comp_value  :: Float64
    data_idx    :: Vector{Int64}
    hdata_idx   :: Vector{Int64}
    left_child  :: Union{Nothing, Node}
    right_child :: Union{Nothing, Node}
    parent      :: Union{Nothing, Node}
    ne          :: NodeEstimator
    locate      :: String
    fm          :: FitMeasure
    Node() = new()
end

# Tree Structure
mutable struct Tree
    root        :: Node
    depth       :: Int64
    leaves      :: Int64
    comp        :: Vector{String}
    mof_gain    :: Vector{Any}
    regions     :: Int64
    ne_type     :: DataType
    fm_type     :: DataType
    group_vars  :: Union{Nothing, Vector{Int64}}
    proj_vars   :: Union{Nothing, Vector{Int64}}
    Tree() = new()
end


#------------------------------------------------------#
### Tree Method -- Prediction ###
#------------------------------------------------------#

# Predict Using Tree
function predict_single_tree(tree, X::AbstractArray, Q::AbstractArray)
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
            Y_hat[r] = predict_grad(node.ne, X, tree.group_vars, tree.proj_vars)[1]
        else
            Y_hat[r] = predict(node.ne, X[:, r])[1]
        end
    end
    return Y_hat
end

# Get Group Membership Using Tree
function predict_membership(tree::Tree, node_groups, Q::AbstractArray)
    grp_mem = zeros(Int64, size(Q)[2])
    for r in range(1, length(grp_mem), step = 1)
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
        term_node_idx = [x == node.locate for x in tree.comp[find_terminal_node(tree.comp)]]
        grp_mem[r] = node_groups[term_node_idx][1]
        
    end
    return grp_mem
end


#------------------------------------------------------#
### Threshold Object Structures ###
#------------------------------------------------------#

# Threshold Object Structure
mutable struct Threshold
    above       :: Int64
    below       :: Int64
    thresh_var  :: Int64
    thresh_val  :: Float64
    l_endpoints :: Vector{Vector{Float64}}
    r_endpoints :: Vector{Vector{Float64}}
end


# Change how nodes, trees, and thresholds are displayed
Base.show(io::IO, f::Tree) = show(io, "$(f.ne_type) Tree of depth $(f.depth) with $(f.leaves) leaves and $(f.regions) regions")
Base.show(io::IO, f::Node) = show(io, "Node split on feature $(f.feature) at value $(f.comp_value)")
Base.show(io::IO, f::Threshold) = show(io, "Region $(f.above) above Region $(f.below) at Q$(f.thresh_var) = $(f.thresh_val) from $(f.l_endpoints) to $(f.r_endpoints).")