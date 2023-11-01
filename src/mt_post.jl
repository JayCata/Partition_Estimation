### Get Standard Errors and Allow for Honest Estimation
function get_insample_se!(tree::Tree, X::AbstractArray, Y::AbstractArray, ne_list, group_list, group_data_indices) 
    term_nodes = tree.comp[find_terminal_node(tree.comp)]
    uniq_groups = unique(group_list)
    n_groups = length(uniq_groups)
    uniq_nes = unique(ne_list)
    for j in range(1, n_groups)
        group_idx = vcat(group_data_indices[group_list .== uniq_groups[j]]...)
        
        fit_se!(uniq_nes[j], X[:,group_idx], Y[group_idx])
      
    end

end

### Getting region level-sample size and purity
function gini_impurity(group_indices, true_data_indices, max = false)
    purity = 0
    if !max
        for true_group_indices in true_data_indices
            purity += (length(intersect(true_group_indices, group_indices))/length(group_indices)) * (1 - length(intersect(true_group_indices, group_indices))/length(group_indices))
        end
    else
        for i in range(1, length(true_data_indices))
            purity += -(1/length(true_group_indices)) * (1 - (1/length(true_group_indices)))
        end
    end
    return purity
end

function entropy(group_indices, true_data_indices, max = false)
    purity = 0
    if !max
        for true_group_indices in true_data_indices
            if length(intersect(true_group_indices, group_indices)) > 0
                purity += -(length(intersect(true_group_indices, group_indices))/length(group_indices)) * log(length(intersect(true_group_indices, group_indices))/length(group_indices))
            end
        end
    else
        for i in range(1, length(true_data_indices))
            purity += -(1/length(true_data_indices)) * log(1/length(true_data_indices))
        end
    end
    return purity
end 

function region_count_and_purity(group_array, data_indices, true_data_indices, pur_measure = entropy)
    counts = zeros(Int64, length(unique(group_array)))
    purities = zeros(Float64, length(unique(group_array)))
    for j in 1:length(unique(group_array))
        group_indices = reduce(vcat, data_indices[group_array .== j])
        counts[j] = length(group_indices)
        purities[j] = pur_measure(group_indices, true_data_indices)    
    end
    max_purity = pur_measure([], true_data_indices, true)
    return counts, purities, max_purity 
end

function get_true_data_indices(true_group_list)
    true_data_indices = Array{Array{Int64,1},1}(undef, length(unique(true_group_list)))
    for j in range(1,length(unique(true_group_list)))
        true_data_indices[j] = findall(x -> x == 1, true_group_list[j][:,1])
    end
    return true_data_indices
end

function plot_region_count_purity(counts, purities; samp_prop = true, nbins = 20, max_impurity = nothing)
    n = sum(counts)
    m = length(counts)
    xlab = "Region-Level Sample Size"
    if samp_prop
        counts = counts ./ n
    end
    pure_regions = (purities .== 0.0)
    impure_regions = (purities .!= 0.0)
    p = scatter(counts[pure_regions], purities[pure_regions], label = "# pure: $(sum(pure_regions))",  color = :blue)
    scatter!(counts[impure_regions], purities[impure_regions], label = "# impure: $(sum(impure_regions))",  color = :red)
    title!("Region Size and Purity (# regions = $m, n = $n)")
    xlabel!("Region-Level Sample Size")
    ylabel!("Region-Level Impurity")
    if !isnothing(max_impurity)
        hline!([max_impurity], linestyle = :dash, label = "Max Impurity")
    end
    return p
end

function make_tree_honest(tree, X, Y, Q; groups = nothing)
    n = length(Y)
    if (isnothing(groups)) && (tree.regions != tree.leaves)
        error("Must specify group structure to make a grouped tree honest.")
    end
    if isnothing(groups)
        groups = [x for x in range(1, tree.regions)]
    end
    term_nodes = tree.comp[find_terminal_node(tree.comp)]
    data_indices  = Array{Array{Int64,1},1}()
    ne_array = Array{Any, 1}(undef, length(term_nodes))
    for j in range(1, length(term_nodes))
        term_node = term_nodes[j]
        Q_bool = ones(Bool, n)
        node = tree.root
        for i in range(1,length(term_node))
            if term_node[i] == '0'
                Q_bool = Q_bool .* (Q[node.feature , :] .<= node.comp_value)
                node = node.left_child
            elseif term_node[i] == '1'
                Q_bool = Q_bool .* (Q[node.feature , :] .> node.comp_value)
                node = node.right_child
            end
        end
      
        push!(data_indices, findall(bool_to_int_index,Q_bool))     
    end
    
    if tree.ne_type <: GradEstimator
        print("nothing yet")
    else
        for group in unique(groups)
            group_term_nodes = term_nodes[groups .== group]
            group_ne = deepcopy(tree.ne_type())
            group_data_idx = vcat(data_indices[groups .== group]...)
            
            fit!(group_ne, X[:, group_data_idx], Y[group_data_idx])
            for term_node in group_term_nodes
                node = tree.root
                for i in range(1,length(term_node))
                    if term_node[i] == '0'
                        node = node.left_child
                    elseif term_node[i] == '1'
                        node = node.right_child
                    end
                end
                node.ne = deepcopy(group_ne)
                node.data_idx = group_data_idx
                
            end
           
            ne_array[groups .== group] .= fill(deepcopy(group_ne), sum(groups .== group))
        end
    end

    return ne_array,data_indices
    
end 