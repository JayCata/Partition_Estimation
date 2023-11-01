#------------------------------------------------------#
### Helper Functions for Joining ### 
#------------------------------------------------------#

# Retrieve Data Indices that Fall into each Terminal Node
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

# Find Best Region Joining
function find_rj(pot_idx, mof_list, num_groups, groups, X::AbstractArray, Y::AbstractArray, ne_type, fm_type)
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
               
                fm_calc_mof!(fm_obj, Y[pot_idx[i, j]], Y_ij_hat, X[:, pot_idx[i, j]], length(Y[pot_idx[i, j]]), ne_obj)
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

# Merge Indices of Leaves in Same Group
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

# Update Tree so New Coefficients are Present
function update_tree(tree::Tree, update_leaves, ne_merge, idx_merge)
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
        node.data_idx = idx_merge
    end
    return tree
end

# Gradient Version of Update Tree
function update_tree_grad(tree::Tree, update_leaves, ne_merge, idx_merge)
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
        node.data_idx = idx_merge
    end
    return tree
end

# Fit Gradient Node Estimators
function convert_to_grad!(tree, X::AbstractArray, Y::AbstractArray)
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
        fit_grad!(node.ne, X[:, node.data_idx], Y[node.data_idx], tree.group_vars, tree.proj_vars)
        X[tree.group_vars, node.data_idx]  = X[tree.group_vars, node.data_idx] .- node.ne.X_demean
        Y[node.data_idx] = Y[node.data_idx] .- X[tree.proj_vars, node.data_idx]' * node.ne.proj_coeff
    end
   
    X = X[tree.group_vars, :]
    return X,Y
      
end 
function calculate_join_mof(obs_idx, Y, X, fm_obj, ne_obj, mof_list, groups, i, j)

    try
        fit!(ne_obj, X[:, obs_idx], Y[obs_idx])
    catch err
        return fm_get_worst_mof!(fm_obj)
    else
        Y_ij_hat  = predict(ne_obj, X[:, obs_idx])
        fm_calc_mof!(fm_obj, Y[obs_idx], Y_ij_hat, X[:, obs_idx], length(Y[obs_idx]), ne_obj)
        return  fm_obj.mof - unique(mof_list[groups .== i])[1] - unique(mof_list[groups .== j])[1]
        
    end
end
function init_classic_mat!(X, Y, criteria_mat, pot_idx, mof_list, num_groups, groups, ne_type, fm_type, adj_mat)
    # Running Objects
    fm_obj = fm_type()
    ne_obj = ne_type()
    if isnothing(adj_mat)
        for i in 2:num_groups
            for j in 1:i
                if pot_idx[i, j] != []
                    criteria_mat[i,j] = calculate_join_mof(pot_idx[i,j], Y, X, fm_obj, ne_obj, mof_list, groups, i, j)
                end
            end
        end
    else
        for i in 2:num_groups
            for j in 1:i
                if pot_idx[i, j] != [] && adj_mat[i, j] == 1
                        criteria_mat[i,j] = calculate_join_mof(pot_idx[i,j], Y, X, fm_obj, ne_obj, mof_list, groups, i, j)
                end
            end
        end
    end
end

function init_param_mat!(X, Y, criteria_mat, pot_idx,  num_groups, groups, ne_group, adj_mat)
    sorted_ne = unique(ne_group[sortperm(groups)])

    # Running Objects
    for i in 2:num_groups
        for j in 1:i
            if pot_idx[i, j] != []
                criteria_mat[i,j] = norm(sorted_ne[j].coeff - sorted_ne[i].coeff)
            end
        end
    end
end

function update_classic_mat!(X, Y, criteria_mat, pot_idx, mof_list, num_groups, groups, ne_type, fm_type, update_index, adj_mat,)
    fm_obj = fm_type()
    ne_obj = ne_type()
    i = update_index
    if isnothing(adj_mat)
        for j in 1:num_groups
            if pot_idx[i, j] != [] 
                criteria_mat[i, j] = calculate_join_mof(pot_idx[i,j], Y, X, fm_obj, ne_obj, mof_list, groups, i, j)
            elseif pot_idx[j, i] != []
                criteria_mat[j, i] = calculate_join_mof(pot_idx[j,i], Y, X, fm_obj, ne_obj, mof_list, groups, j, i)
            end
        end
    else
        for j in 1:num_groups
            if pot_idx[i, j] != [] && adj_mat[i, j] == 1
                criteria_mat[i, j] = calculate_join_mof(pot_idx[i,j], Y, X, fm_obj, ne_obj, mof_list, groups, i, j)
            elseif pot_idx[j, i] != [] && adj_mat[j, i] == 1
                criteria_mat[j, i] = calculate_join_mof(pot_idx[j,i], Y, X, fm_obj, ne_obj, mof_list, groups, j, i)
            end
        end
    end
end

function update_param_mat!(X, Y, criteria_mat, pot_idx, num_groups, groups, ne_group, update_index, adj_mat)
    sorted_ne = unique(ne_group[sortperm(groups)])
    i = update_index
    for j in 1:num_groups
        if pot_idx[i, j] != []
            criteria_mat[i,j] = norm(sorted_ne[i].coeff - sorted_ne[j].coeff)
        elseif pot_idx[j, i] != []
            criteria_mat[j,i] = norm(sorted_ne[j].coeff - sorted_ne[i].coeff)
        end
    end
end

function find_best_join!(X::AbstractArray, Y::AbstractArray, criteria_mat, pot_idx, mof_list, groups, ne_type, fm_type)
    ~, mg = findmin(criteria_mat)

    mg_1 = mg[2]
    mg_2 = mg[1]
    fm_obj_opt = fm_type()
    ne_obj_opt = ne_type()
    fit!(ne_obj_opt, X[:, pot_idx[mg_2, mg_1]], Y[pot_idx[mg_2, mg_1]])
    Y_ij_hat  = predict(ne_obj_opt, X[:, pot_idx[mg_2, mg_1]])
    fm_calc_mof!(fm_obj_opt, Y[pot_idx[mg_2, mg_1]], Y_ij_hat, X[:, pot_idx[mg_2, mg_1]], length(Y[pot_idx[mg_2, mg_1]]), ne_obj_opt)
    fm_obj_opt.mof_gain = fm_obj_opt.mof - unique(mof_list[groups .== mg_2])[1] - unique(mof_list[groups .== mg_1])[1]
    return mg_1, mg_2, fm_obj_opt, ne_obj_opt
end

function compare_paths(info_i, info_j)
    count = 0
    for i in range(1,size(info_i)[1])
        for j in range(1,size(info_j)[1])
            feat_match = (info_i[i,1] ==  info_j[j,1])
            iaj_val = (info_i[i,2] >=  info_j[j,2])
            iaj_bool = (info_i[i,3] >  info_j[j,3])
            ibj_val = (info_i[i,2] <=  info_j[j,2])
            ibj_bool = (info_i[i,3] <  info_j[j,3])
            count += feat_match * (iaj_val * iaj_bool + ibj_val * ibj_bool)
        end
    end
    if count > 1
        return false
    else
        return true
    end
end

function init_adjacency_matrix(tree)
    term_nodes = tree.comp[find_terminal_node(tree.comp)]
    T = length(term_nodes)
    adj_mat = sparse(zeros(Bool, T, T))
    node_info_list = []
    for i in range(1, T)
        str_i = term_nodes[i]
        tups_i = Array{Any}(undef, 0, 3)
        node_i = tree.root
        for s in range(1, length(str_i))
            tups_i = [tups_i; [node_i.feature node_i.comp_value string(str_i[s])]]
            if str_i[s] == '0'
                node_i = node_i.left_child
            elseif str_i[s] == '1'
                node_i = node_i.right_child
            end
            
        end
       
       push!(node_info_list, tups_i)
    end

    for i in range(1, T)
        for j in range(i+1, T)
            if compare_paths(node_info_list[i], node_info_list[j])
                adj_mat[j,i] = 1
            end
            
        end
    end
    return adj_mat
end

function update_adjacency_matrix!(adj_mat, mg_1, mg_2)
    T = size(adj_mat)[1]
    temp_row = vcat(adj_mat[mg_2, :][1:mg_1-1], zeros(T-mg_1+1));
    temp_col = adj_mat[:, mg_2] + vcat(zeros(mg_1), adj_mat[mg_2, :][mg_1+1:end]);
    adj_mat[mg_1, :] = min.(1, adj_mat[mg_1, :] + temp_row);
    adj_mat[:, mg_1] = min.(1, adj_mat[:, mg_1] + temp_col);
end
#------------------------------------------------------#
### Joining Functions ### 
#------------------------------------------------------#
function region_join(tree::Tree, X::AbstractArray, Y::AbstractArray; join_criteria = "classic" , contiguous = false)
    # Start timer
    s_time = time()
    if tree.ne_type <: GradEstimator
        X = deepcopy(X)
        Y = deepcopy(Y)
    end

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
    # find terminal nodes (leaves)
    term_node_bool = find_terminal_node(tree.comp)
    T = sum(term_node_bool)
    leaf_comp = tree.comp[term_node_bool]

    # If this is a gradient estimation, select correct variables
    if tree.ne_type <: GradEstimator
        X, Y = convert_to_grad!(tree, X, Y)
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

    # Initialize criteria matrix
    criteria_mat = fill(Inf64, num_groups, num_groups)
    init = true
    adj_mat = nothing
    if contiguous
        adj_mat = init_adjacency_matrix(tree)
    end
    mg_1 = 0
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
       

        if init == true
            if join_criteria == "classic"
                init_classic_mat!(X, Y, criteria_mat, pot_idx, mof_leaves, num_groups, groups, tree.ne_type, tree.fm_type, adj_mat)
            elseif  join_criteria == "param_dist"
                init_param_mat!(X, Y, criteria_mat, pot_idx, num_groups, groups, ne_group, adj_mat)
            end
            init = false
        else    
            if join_criteria == "classic"
               update_classic_mat!(X, Y, criteria_mat, pot_idx, mof_leaves, num_groups, groups, tree.ne_type, tree.fm_type, mg_1, adj_mat)
            elseif  join_criteria == "param_dist"
                update_param_mat!(X, Y, criteria_mat, pot_idx, num_groups, groups, ne_group, mg_1, adj_mat)
            end
        end
        
        # Find best join and output the two regions that are being joined
        mg_1, mg_2, fm_update, ne_merge = find_best_join!(X, Y, criteria_mat, pot_idx, mof_leaves, groups, tree.ne_type, tree.fm_type)
        
        # Update Groups vector (mg_2 groups are now mg_1)
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
            tree = deepcopy((update_tree_grad(tree, update_leaves, ne_merge, pot_idx[mg_2, mg_1])))
        else
            for x in updated_indices
                ne_group[x] = ne_merge
            end
            tree = deepcopy((update_tree(tree, update_leaves, ne_merge, pot_idx[mg_2, mg_1])))
        end

        # Update group numberings so any single group number is not larger than the number of groups there are
        if sum(groups .> mg_2) != 0
            for i in 1:length(groups)
                if groups[i] > mg_2
                    groups[i] = groups[i] - 1
                end
            end     
        end
        criteria_mat = criteria_mat[1:size(criteria_mat, 1).!= mg_2,  1:size(criteria_mat, 1) .!= mg_2]
        if !isnothing(adj_mat)
            update_adjacency_matrix!(adj_mat, mg_1, mg_2)
            adj_mat = adj_mat[1:size(adj_mat, 1).!= mg_2,  1:size(adj_mat, 1) .!= mg_2]
           
        end
        # Update Tree Parameters to reflect grouping and push that tree to treelist 
        tree.regions = num_groups
        push!(tree_list, deepcopy(tree))
        push!(mof_gain_list, deepcopy(fm_update.mof_gain))
        push!(ne_list, deepcopy(ne_group))
        push!(group_list, deepcopy(groups))
        
        
    end
    println("time for joining one ", join_criteria, " tree: ", time()-s_time)
    normalized_mof = cumsum(mof_gain_list)
    return tree_list, group_list, ne_list, normalized_mof, indices
end

function rj_full_tree(tree::Tree, X::AbstractArray, Y::AbstractArray; join_criteria = "classic", contiguous = false, group_vars = nothing, full_reg = true, num_alpha = 60, alpha_range = (.5,10))
    n = length(Y)
    highest_alpha, lowest_alpha = alpha_range
    if tree.ne_type <: GradEstimator && (group_vars == nothing || group_vars == [x for x in range(1,n)])
        error("Need to Specify group_vars when using estimate of type GradEstimator")
    end 
    tree.group_vars = group_vars
    if !isnothing(group_vars)
        proj_vars = deleteat!([x for x in range(1,size(X,1))], findall(x -> x in group_vars, [x for x in range(1,size(X,1))]))
        tree.proj_vars = proj_vars
    end
    tree_list, group_list, ne_list, normalized_mof, indices = region_join(tree::Tree, X::AbstractArray, Y::AbstractArray, join_criteria = join_criteria, contiguous = contiguous)
    ## Standard
    #regularize_final = [x for x in range(0, stop = 1.1*(maximum(mof_gain_list)), length = 2 * length(tree_list))]
    #regularize_final = sort(unique(vcat(regularize_final, mof_gain_list)))
    #penalized_term = [x * y.regions for x = regularize_final, y in tree_list]
    #regularize_final = regularize_final./n
    ## 
    if full_reg
        highest_alpha = 1.01*maximum(diff(normalized_mof))/((var(Y)) * log(n))
        lowest_alpha = highest_alpha/1000
    end 
    regularize_final = [x for x in range(lowest_alpha, highest_alpha, length = num_alpha)]
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    treefinalindex = Vector{Int64}()
    for i in 1:length(regularize_final)
        push!(treefinalindex, findmin(normalized_mof .+ penalized_term[i,:])[2])
    end
    
    return tree_list, group_list, ne_list, treefinalindex, regularize_final, indices
end

function val_rj(tree, regularize_final, X::AbstractArray, Y::AbstractArray, join_criteria, contiguous, group_vars)
    n = length(Y)
    if tree.ne_type <: GradEstimator && (group_vars == nothing || group_vars == [x for x in range(1,n)])
        error("Need to Specify group_vars when using estimated of type GradEstimator")
    end 
    tree.group_vars = group_vars
    if !isnothing(group_vars)
        proj_vars = deleteat!([x for x in range(1,size(X,1))], findall(x -> x in group_vars, [x for x in range(1,size(X,1))]))
        tree.proj_vars = proj_vars
    end
    tree_list, group_list, ne_list, normalized_mof, indices = region_join(tree::Tree, X::AbstractArray, Y::AbstractArray, join_criteria = join_criteria, contiguous = contiguous)
    
    ## Standard
    #regularize_final = n * regularize_final 
    #penalized_term = [x * y.regions  for x = regularize_final, y in tree_list]
    ## 
    final_tree_list = Vector{Tree}()
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(final_tree_list, tree_list[findmin(normalized_mof .+ penalized_term[i,:])[2]])
    end
    return final_tree_list
end

function nfold_cross_val_rj(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, regularize_final, minobs, ne_type, fm_type; nfolds=20, join_criteria = "classic", contiguous = false, group_vars = nothing, early_stop = false)
   
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
        tree = grow_tree(X[:,train_ind], Y[train_ind], Q[:, train_ind], feature_idx, minobs, ne_type, fm_type, early_stop = early_stop)
        # Prune tree based on full tree alpha levels
        val_trees = val_rj(tree, regularize_final, X[:, train_ind], Y[train_ind], join_criteria, contiguous, group_vars)
        nval = length(Y[train_ind])
        predict_val(x) = predict_single_tree(x, X[:,fold_vec[i]], Q[:,fold_vec[i]])
        Y_hat_list = predict_val.(val_trees)
        for j in 1:length(regularize_final)
            fm_obj = fm_type()
            fm_calc_mof!(fm_obj, Y[fold_vec[i]], Y_hat_list[j], X[:,train_ind], 0, tree.root.ne) # last argument is filler
           
            mof_tally[i,j] = fm_obj.mof
        end
    end
    mof_totals= sum(mof_tally,dims = 1)[1,:]
    return findall(x -> x == minimum(mof_totals), mof_totals)
end

#------------------------------------------------------#
### Old Joining Functions ### 
#------------------------------------------------------#
function region_join_old(tree::Tree, X::AbstractArray, Y::AbstractArray)
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
    println("time for joining one tree: ", time()-s_time)
    normalized_mof = cumsum(mof_gain_list)
    return tree_list, group_list, ne_list, normalized_mof, indices
end

function rj_full_tree_old(tree::Tree, X::AbstractArray, Y::AbstractArray)
    n = length(Y)
    tree_list, group_list, ne_list, normalized_mof, indices = region_join_old(tree::Tree, X::AbstractArray, Y::AbstractArray)
    ## Standard
    #regularize_final = [x for x in range(0, stop = 1.1*(maximum(mof_gain_list)), length = 2 * length(tree_list))]
    #regularize_final = sort(unique(vcat(regularize_final, mof_gain_list)))
    #penalized_term = [x * y.regions for x = regularize_final, y in tree_list]
    #regularize_final = regularize_final./n
    ##  
    regularize_final = [x for x in range(.5, 8, length = 40)]
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    treefinalindex = Vector{Int64}()
    for i in 1:length(regularize_final)
        push!(treefinalindex, findmin(normalized_mof .+ penalized_term[i,:])[2])
    end
    
    return tree_list, group_list, ne_list, treefinalindex, regularize_final, indices
end


# Validation region joining, taking regularization paramaters as given
function val_rj_old(tree, regularize_final, X::AbstractArray, Y::AbstractArray)
    n = length(Y)
    tree_list, group_list, ne_list, normalized_mof, indices = region_join_old(tree::Tree, X::AbstractArray, Y::AbstractArray)
    
    ## Standard
    #regularize_final = n * regularize_final 
    #penalized_term = [x * y.regions  for x = regularize_final, y in tree_list]
    ## 
    final_tree_list = Vector{Tree}()
    penalized_term = [x * y.regions * log(n) * var(Y) for x = regularize_final, y in tree_list]
    for i in 1:length(regularize_final)
        push!(final_tree_list, tree_list[findmin(normalized_mof .+ penalized_term[i,:])[2]])
    end
    return final_tree_list
end

# Cross Validation for region joining 
function nfold_cross_val_rj_old(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, regularize_final, nfolds, minobs, ne_type, fm_type)
   
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
        val_trees = val_rj_old(tree, regularize_final, X[:, train_ind], Y[train_ind])
        nval = length(Y[train_ind])
        predict_val(x) = predict_single_tree(x, X[:,fold_vec[i]], Q[:,fold_vec[i]])
        Y_hat_list = predict_val.(val_trees)
        for j in 1:length(regularize_final)
            fm_obj = fm_type()
            #fm_calc_mof!(fm_obj, Y[fold_vec[i]], Y_hat_list[j])
            mof_tally[i,j] = fm_obj.mof
        end
    end
    mof_totals= sum(mof_tally,dims = 1)[1,:]
    return findall(x -> x == minimum(mof_totals), mof_totals)
end