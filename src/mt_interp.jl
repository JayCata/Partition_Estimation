

#------------------------------------------------------#
### Interpretation Algorithims ###
#------------------------------------------------------#

# Retrieves all Hyperrectangles from Tree
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

# Compares Thresholds
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

# Gets Largest Hyperrectangle
function get_largest_hyperrectangles(groups, Q::AbstractArray, γl, γu)
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

# Interpret Borders as Thresholds
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

# Get Model Information
function get_thresholds(tree, groups)
    Qindexs, ne, γls, γus = get_tree_hyperrectangles(tree)
    Qindex, γl, γu = get_largest_hyperrectangles(groups, Qindexs, γls, γus)
    return get_thresholds_from_borders(γl, γu), Qindex
end