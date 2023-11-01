# Find All Terminal Nodes
function find_terminal_node(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 1) 
    end
    return term
end

# Label Data Points Based on Group Membership
function data_groups(tree::Tree, Q::AbstractArray, group_labels)
    groups = Array{Int64}(undef, 0)
    terminal_nodes = tree.comp[find_terminal_node(tree.comp)]
    
    for r in range(1, size(Q,2), step = 1)
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

# Take n Fold Sample
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

# BIC Function
function BIC(joined_trees, Y::AbstractArray, X::AbstractArray, Q_tilde::AbstractArray)
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
        BICs[counter] = n * log(sse/n) + mhat* d * log(n) 
        counter += 1
    end
    return argmin(BICs)
end

#
function model_select(joined_trees, Y::AbstractArray, X::AbstractArray, Q_tilde::AbstractArray, penalty_scaler = 2)
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
        BICs[counter] =sse + mhat * d * log(n)  * var(Y) * penalty_scaler
        counter += 1
    end
    return argmin(BICs)
end

