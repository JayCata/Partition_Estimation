#------------------------------------------------------#
### Minor Helper Functions for Tree Growing ### 
#------------------------------------------------------#

# Creates Split Object 
function create_split_obj(feature, min_val, max_val, split_val, left_idx, right_idx, hleft_idx, hright_idx, left_ne, right_ne, left_fm, right_fm)
    new_so = SplitObj()
    new_so.feature = feature
    new_so.min = min_val
    new_so.max = max_val
    new_so.split_val = split_val
    new_so.left_idx = left_idx
    new_so.right_idx = right_idx
    new_so.hleft_idx = hleft_idx
    new_so.hright_idx = hright_idx
    new_so.left_ne = left_ne
    new_so.right_ne = right_ne
    new_so.left_fm = left_fm
    new_so.right_fm = right_fm
    return new_so
end

# Convert Boolean Index to Integer Index
function bool_to_int_index(a)
    a == 1
end

# Adds Node to Queue
function queue_compute_nodes!(queue::Vector{Node}, node::Node)
    push!(queue, node)
end

# Count the Number of Leaves
function count_leaves(comps)
    numnodes = length(comps)
    term = zeros(Bool, numnodes)
    for j in 1:numnodes
        term[j] = (sum(startswith.(comps, comps[j])) == 1) 
    end
    return sum(term)
end


#------------------------------------------------------#
### Major Helper Functions for Tree Growing ### 
#------------------------------------------------------#

# Finds Split that Greedily Optimizes Measure of Fit
function find_best_split(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, Q_honest::AbstractArray, feature_idx, minobs, ne_type, fm_type)
    # Initialize best variables
    best_feature_idx = -1
    best_split_val = -Inf64
    best_min_val = -Inf64
    best_max_val = Inf64
    best_lidx = nothing
    best_ridx = nothing
    best_hlidx = nothing
    best_hridx = nothing
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

    if (length(Y) >= 2 * minobs) && (size(Q_honest, 2) >= 2 * minobs)
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
                        fm_calc_mof!(l_fm, Y[lidx], Y_hat_l, X[:,lidx],  length(Y), l_ne)
                        fm_calc_mof!(r_fm, Y[ridx], Y_hat_r, X[:,ridx], length(Y), r_ne)
                        
                        # Check to see if best split
                        if fm_compare_grow(l_fm, r_fm, best_l_fm, best_r_fm)
                            best_feature_idx = i
                            best_split_val = split_val
                            best_l_fm = deepcopy(l_fm)
                            best_r_fm = deepcopy(r_fm)
                            best_min_val = min_val
                            best_max_val = max_val
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
        if nfeatures !== 1
            best_hlidx = Q_honest[best_feature_idx,:] .<= best_split_val
            best_hridx = Q_honest[best_feature_idx,:] .> best_split_val
        else
            best_hlidx = reshape(Q_honest, length(Q_honest)) .<= best_split_val
            best_hridx = reshape(Q_honest, length(Q_honest)) .> best_split_val
        end

        if nfeatures !== 1
            best_lidx = Q[best_feature_idx,:] .<= best_split_val
            best_ridx = Q[best_feature_idx,:] .> best_split_val
        else
            best_lidx = reshape(Q, length(Q)) .<= best_split_val
            best_ridx = reshape(Q, length(Q)) .> best_split_val
        end
        
        
        split_obj.feature = feature_idx[best_feature_idx]
        best_lidx = findall(bool_to_int_index, best_lidx)
        best_ridx = findall(bool_to_int_index, best_ridx)
        best_hlidx = findall(bool_to_int_index, best_hlidx)
        best_hridx = findall(bool_to_int_index, best_hridx)
        split_obj = create_split_obj(feature_idx[best_feature_idx], best_min_val, best_max_val, best_split_val, best_lidx, best_ridx, best_hlidx, best_hridx, best_l_ne, best_r_ne, best_l_fm, best_r_fm)
        return split_obj
    end
    return split_obj
end

# Convert Split Object to Node
function compute_node!(tree::Tree, node::Node, X::AbstractArray, Y::AbstractArray, Q::AbstractArray, Q_honest::AbstractArray, feature_idx, minobs, ne_type, fm_type, early_stop)
    # last three params if split needs to be restarted
    @views split_obj = find_best_split(X[:,node.data_idx], Y[node.data_idx], Q[:,node.data_idx], Q_honest[:,node.hdata_idx], feature_idx, minobs, ne_type, fm_type)
    
    
    node.fm.mof_gain = nothing 
    node.left_child = nothing
    node.right_child = nothing

    if !isinf(split_obj.split_val)
        node.fm.mof_gain = node.fm.mof - split_obj.left_fm.mof - split_obj.right_fm.mof
        if stop_early(node, split_obj, X, Y, node.fm) || (early_stop == false)
            left_node = Node()
            right_node = Node()
            left_node.data_idx = node.data_idx[split_obj.left_idx]
            right_node.data_idx = node.data_idx[split_obj.right_idx]
            left_node.hdata_idx = node.hdata_idx[split_obj.hleft_idx]
            right_node.hdata_idx = node.hdata_idx[split_obj.hright_idx]
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
            node.feature = split_obj.feature
            node.comp_value = split_obj.split_val
            
            return left_node, right_node, node.fm.mof_gain
        end
    end
  
    return nothing, nothing, nothing
end

# Create the Roote Node
function create_root_node!(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, Q_honest::AbstractArray, feature_idx, minobs, ne_type, fm_type, early_stop)
    split_obj = find_best_split(X, Y, Q, Q_honest, feature_idx, minobs, ne_type, fm_type)
    
    queue = Vector{Node}()
    
    if !isinf(split_obj.split_val)
        node = Node()
        node.feature = split_obj.feature
        node.comp_value = split_obj.split_val

        node.data_idx = collect(1:size(Q)[2])
        node.ne = ne_type()
        node.fm = fm_type()
        fit!(node.ne, X, Y)
        Y_hat = predict(node.ne, X)
        fm_calc_mof!(node.fm, Y, Y_hat, X,  length(Y), node.ne)
        node.fm.mof_gain = node.fm.mof - split_obj.left_fm.mof - split_obj.right_fm.mof
        
        if stop_early(node, split_obj, X, Y, node.fm) || (early_stop == false)
            left_node = Node()
            left_node.feature = -1
            left_node.left_child = nothing
            left_node.right_child = nothing
            left_node.ne= split_obj.left_ne
            left_node.comp_value = NaN
            left_node.data_idx = split_obj.left_idx
            left_node.hdata_idx = split_obj.hleft_idx
            left_node.locate = "0"
            left_node.fm = split_obj.left_fm
            
            right_node = Node()
            right_node.feature = -1
            right_node.left_child = nothing
            right_node.right_child = nothing
            right_node.ne = split_obj.right_ne
            right_node.comp_value = NaN
            right_node.data_idx = split_obj.right_idx
            right_node.hdata_idx = split_obj.hright_idx
            right_node.locate = "1"
            right_node.fm = split_obj.right_fm
            
            
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

#------------------------------------------------------#
### Tree Growing Algorithm ### 
#------------------------------------------------------#
# Grows Tree
function grow_tree(X::AbstractArray, Y::AbstractArray, Q::AbstractArray, feature_idx, minobs, ne_type = ols, fm_type = sse; Q_honest = nothing, early_stop = false)
    comp = [""]
    
    s_time = time()
    if ndims(Q) == 1
        Q = reshape(Q, 1, length(Q))
    end

    if !isnothing(Q_honest)
        if ndims(Q_honest) == 1
            Q_honest= reshape(Q_honest, 1, length(Q_honest))
        end
    end

    
    
    tree = Tree()
    depth = 0 
    @views X_view = X
    @views Y_view = Y
    @views Q_view = Q
    @views Q_honest_view = isnothing(Q_honest) ? Q : Q_honest

    root, queue = create_root_node!(X_view, Y_view, Q_view, Q_honest_view, feature_idx, minobs, ne_type, fm_type, early_stop)
    if length(queue) > 0
        comp = vcat(comp, "0", "1")
    end
    
    mof_gain_list = [root.fm.mof_gain]
    
    while length(queue) > 0
        node = popfirst!(queue)
        left_node, right_node, mof_gain = compute_node!(tree, node, X_view, Y_view, Q_view, Q_honest_view, feature_idx, minobs, ne_type, fm_type, early_stop)
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