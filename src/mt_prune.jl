#------------------------------------------------------#
### Helper Functions for Pruning ### 
#------------------------------------------------------#

# Find All Nodes Resulting in a Terminal Split
function init_terminal_split(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 3) 
    end
    return term
end

# Find Optimal Prune
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


#------------------------------------------------------#
### Pruning Functions ### 
#------------------------------------------------------#

# Prune Tree
function prune(tree::Tree, Y::AbstractArray)
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
    
    println("time for pruning one tree: ", time()-s_time)
    # Intiialze indices that point to which tree is optimal for a given regularization parameter
    
    # Get normalized mof from the gains 
    normalized_mof = cumsum(mof_gain_list)
    return tree_list, normalized_mof
end

function prune_full_tree(tree::Tree,Y::AbstractArray)
    tree_list, normalized_mof = prune(tree, Y)
    n = length(Y)
    # Make grid of regularization values
    regularize_final = [x for x in range(.5, 3, length = 20)]
    tree_final_index = Vector{Int64}()
    # This line adds the specific regularization values to the grid as well 
    # regularize_final = sort(unique(vcat(regularize_final, mof_gain))) 

    # Create penaliztaion term as matrix
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(tree_final_index, findmin(normalized_mof .+ penalized_term[i,:])[2])
    end
    
    return tree_list, tree_final_index, regularize_final
end

# Validation Pruning
function prune_val(tree::Tree, regularize_final, Y::AbstractArray)
    n = length(Y)
    tree_list, normalized_mof = prune(tree, Y)
    # Initialize list of trees
    final_tree_list = Vector{Tree}()

    # Create penalization term as matrix
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(final_tree_list, tree_list[findmin(normalized_mof .+ penalized_term[i,:])[2]])
    end
    
    return final_tree_list
end

# Nfold Cross Validation Pruning
function nfold_cross_val_prune(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, regularize_final, minobs, ne_type, fm_type; nfolds = 20)
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
            fm_calc_mof!(fm_obj, Y[fold_vec[i]], Y_hat_list[j], X, length(Y), ne_type())
            mof_tally[i,j] = fm_obj.mof
        end
    end
    mof_totals= sum(mof_tally,dims = 1)[1,:]
    return findall(x -> x == minimum(mof_totals), mof_totals)
end