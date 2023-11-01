using Distributions
using Random
using LinearAlgebra
using Statistics
using JLD2, FileIO
using BenchmarkTools
using Plots, StatsPlots
using Combinatorics
using FixedEffectModels
using DataFrames
using Suppressor
# Set Working Dorectory and include ModularTrees.jl


cd(@__DIR__)
include("..\\..\\src\\part_est.jl")
Random.seed!(456)
function generate_trend!(panel_data, params, num_I, T, Tstar, Γ)
    eps_dis = Normal(0, .5)
    
    Yold = 1 .+ [(s/num_I) for s in range(1,num_I)] + rand(eps_dis, num_I)
    Y = Yold
    Ydiff = fill(Inf, num_I)
    for t in range(2,T)
      
        Ynew = Yold + sum(params[1]'  .* panel_data["Γ"][(t-2) * num_I + 1:num_I*(t-1),:], dims = 2) + rand(eps_dis, num_I) .+ ((t >= Tstar) * (Γ * params[2] * (t - Tstar + 1)) .* (panel_data["treatment"][(t-2) * num_I + 1:num_I*(t-1),:]) )
        
        Y = vcat(Y, Ynew)
        Ydiff = vcat(Ydiff, Ynew - Yold)
        Yold = Ynew
    end
    
    panel_data["Y"] = Y
    panel_data["Ydiff"] = Ydiff
    panel_data["X"] = reduce(hcat,[[x == s for x in panel_data["t"]] for s in unique(panel_data["t"])])
    
end

function complex_partition(q)
    if (round(q[1]; digits=0) + round(q[2]; digits=0))/1.5 < -1
        return 1
    elseif (round(q[1]; digits=0) + round(q[2]; digits = 0))/1.5 > 1
        return 3
    else
        return 2
    end
end

# Generate Data given # of individuals (I),  # of true threshold variables (k), # of potential threshold variables (l), T total number of periods
function generate_data(num_I, T, k, l) 
    Q = rand(Uniform(-2,2), num_I, k)
    Q_tilde = Q
    # Noisy Vars
    X = []
    if l > k
        X = rand(Normal(0, 2), num_I, l-k)
        Q_tilde = hcat(Q, X)
    end
    


    Γ1 = [complex_partition(q) == 1 for q in eachrow(Q)]
    Γ2 = [complex_partition(q) == 2 for q in eachrow(Q)]
    Γ3 = [complex_partition(q) == 3 for q in eachrow(Q)]
    Γ = hcat(Γ1, Γ2, Γ3)
    # Treatment assignment
    treat = rand.(Bernoulli.(Γ1 * .8 + Γ2 * .5 + Γ3 * .2))

    # Create 
    # Create Panel Data
    ind_data = hcat([i for i in range(1,num_I)], Q_tilde, treat, Γ1, Γ2, Γ3)
    panel_data =  reduce(hcat, [vcat([y], x) for x in eachrow(ind_data), y in [t for t in range(1,T)]])'
    panel_data = Dict(
        [("t", panel_data[:,1]),
        ("i", Int.(panel_data[:,2])),
        ("Q", panel_data[:,3:3+k-1]),
        ("Q_tilde", panel_data[:,3:3+l-1]),
        ("treatment", panel_data[:,3+l]),
        ("Γ", panel_data[:, 3+l+1:end])]
    )
    return panel_data, Γ
end

function dict_to_df(groups, data_indices, data, Tstar)
    individual_partition = []
    for j in unique(groups)
        push!(individual_partition, Int.(unique(data["i"][reduce(vcat, data_indices[groups .== j])])))
    end

    
    
    data["post_Tstar"]  = data["t"] .>= Tstar
    data["trend_var"] = data["t"] .- Tstar .+ 1
    treat_temp = data["treatment"]

    df = DataFrame(Y = data["Y"][:,1],
                        Ydiff = data["Ydiff"][:,1],
                        i = data["i"],
                        t = data["t"],
                        treat = data["treatment"],
                        post_Tstar = data["post_Tstar"],
                        trend_var = data["trend_var"] )

    return individual_partition, df
end

function bt_sample(panel_data; n_samp = nothing, replace = true)
    if isnothing(n_samp)
        n_samp = length(panel_data["Y"])
    end
    

    bt_idx = rand(Categorical((1/length(panel_data["Y"])) * ones(length(panel_data["Y"]))), n_samp)
    bt_data = Dict()
    bt_data["data_idx"] = bt_idx
    for (key, value) in panel_data
        if typeof(value) <: AbstractVector
            bt_data[key] = value[bt_idx]
        else
            bt_data[key] = value[bt_idx, :]
        end
    end
    return bt_data
end

function match_bt_to_fs(idx_1, group_1, permute_1, idx_2, group_2, permute_2)
    g1_idx = [vcat([permute_1[idx] for idx in idx_1[group_1 .== j]]...) for j in unique(group_1)]
    g2_idx = [vcat([permute_2[idx] for idx in idx_2[group_2 .== j]]...) for j in unique(group_2)]
    return [sum(g1 ∩ g2)/sum(unique(g1)) for g1 in g1_idx, g2 in g2_idx]


end

function match_fs_to_main(main_grp_idx, self_grp_idx, uniq_main, uniq_self)
    match_mat =  [sum((main_grp_idx .== l) .* (self_grp_idx .== k))/sum((self_grp_idx .== k)) for k in uniq_self, l in uniq_main]
    return match_mat
end

function calculate_boot_se(estimator_vec)
    sample_matrix =  [estimator_vec[b][j][t] for b in range(1, length(estimator_vec)), j in range(1 ,length(estimator_vec[1])), t in range(1,length(estimator_vec[1][1]))]
    boot_means = mean(sample_matrix, dims = 1)
    boot_se = sqrt.(mean((sample_matrix -  repeat(boot_means, outer = [length(estimator_vec),1,1])) .^ 2, dims = 1))
    return boot_se[1,:,:]
end

function calculate_empirical_CI(estimator_vec, α)
    est_list = [[theta[b][j] for b in range(1, length(estimator_vec))] for j in range(length(estimator_vec[1]))]
    return quantile.(est_list, [(α/2), 1-(α/2)])
end

function convert_estim_to_mat(estimator_vec)
    return vcat([estimator_vec[j][1]' for j in range(1, length(estimator_vec))]...)
end

function convert_se_to_mat(estimator_vec)
    return vcat([estimator_vec[j][2]' for j in range(1, length(estimator_vec))]...)
end


function add_panel_data!(panel_df)

end
# Simulation Invariant Parameters
# Number of TIme Periods
T = 6
# Treatment Time Period
Tstar = 5

# Number of true threshold variables
k = 2

# Number of potential threshold variables
l = 2

# Features algorithm considers (all)
feature_idx = [x for x in range(1, l)]

# True Paramaeters
params = ([1, 2, 3], [1, .5, .2])

# Various Numbers of Individuals
num_I = 4000



# Number of Simulations  for Coverpage Probability 
num_sims = 200

# Store Group-Level Trend Estimates
trend_est = Array{Any}(nothing, num_sims)

# Store Group-Level Treatment Estimates
treat_est = Array{Any}(nothing, num_sims)

# Store Aggregate Estimate
probs = Array{Any}(nothing, num_sims)
treat_agg_est = Array{Any}(nothing, num_sims)
treat_agg_ses = Array{Any}(nothing, num_sims)
agg_var = Array{Any}(nothing, num_sims)

# The model selected tree we use to choose the number of groups
main_partition = []
main_groups = []

# So we can references these
match_main_permute = []
group_regs = []
s=1

for s in range(1, num_sims)
    println("Simulation number $(s)")
    # Generate data & Trends
    panel_data, Γ = generate_data(num_I, T, k, l)
    generate_trend!(panel_data, params, num_I, T, Tstar, Γ)
    
    # Set minimum number of observations in leaf
    minobs = sqrt(T * num_I)

    # Grow Tree
    pre_trend_ind = 1 .< panel_data["t"] .< Tstar
    full_tree = MT.grow_tree(panel_data["X"][pre_trend_ind,2:(Tstar-1)]', panel_data["Ydiff"][pre_trend_ind,:], panel_data["Q_tilde"][pre_trend_ind, :]', feature_idx, minobs, MT.ols, MT.sse; early_stop = true)

    # Join Tree
    joined_trees, group_list, joined_θlist, joined_trees_index, joined_αs, joined_data_indices = MT.rj_full_tree(deepcopy(full_tree), panel_data["X"][pre_trend_ind,2:Tstar-1]', panel_data["Ydiff"][pre_trend_ind, :]; contiguous = true)

    # Cross validate for main tree or match num regions to main tree
    if s == 1
        best_tree_idx = [MT.model_select(joined_trees, panel_data["Ydiff"][pre_trend_ind, :], panel_data["X"][pre_trend_ind, 2:(Tstar-1)]', panel_data["Q_tilde"][pre_trend_ind, :]')]
    else
        best_tree_idx = [x.regions == main_partition.regions for x in joined_trees]
    end
    
    

    # Get Best Performing Tree Object, grouping of leaves, and group-level estimates
    best_joined_tree = deepcopy(unique(joined_trees[best_tree_idx]))[1]
    best_joined_groups = deepcopy(unique(group_list[best_tree_idx]))[1]
    best_joined_θ = deepcopy(unique(joined_θlist[best_tree_idx]))[1]

    # Create main partition or match to main
    if s == 1
        main_partition = best_joined_tree
        main_groups = best_joined_groups
        match_main_permute = [x for x in range(1, best_joined_tree.regions)]
        println("Best Number of Regions $(best_joined_tree.regions)")
    else
        grp_mem_in_main = MT.predict_membership(main_partition, main_groups, panel_data["Q_tilde"][pre_trend_ind, :]')
        grp_mem_in_self = MT.predict_membership(best_joined_tree, best_joined_groups, panel_data["Q_tilde"][pre_trend_ind, :]')
        match_main_permute = unique(main_groups)[[cp[1] for cp in argmax(match_fs_to_main(grp_mem_in_main, grp_mem_in_self, unique(main_groups), unique(best_joined_groups)), dims =1)]'][:,1] 
    end

    MT.get_insample_se!(best_joined_tree, panel_data["X"][pre_trend_ind, 2:(Tstar-1)]',  panel_data["Ydiff"][pre_trend_ind, :], best_joined_θ, best_joined_groups, joined_data_indices) 
    trend_est[s] =  [[x.coeff, x.se] for x in unique(best_joined_θ)[unique(best_joined_groups)]][match_main_permute]

    individual_partition, panel_df = dict_to_df(best_joined_groups, joined_data_indices, panel_data, Tstar)
    individual_partition = individual_partition[match_main_permute]

        
    treatment_temp = []
    temp_treats = []
    temp_prob_num = []
    temp_prob_den = mean((panel_data[ "treatment"] .== 1))
    panel_df[:, "j"] = zeros(Int64,nrow(panel_df))
    for (idx, j) in enumerate(individual_partition)
        panel_df[:, "j"] += Int.(idx .* (panel_df[:, "i"] .∈ Ref(j)))
    end

    for (idx, j) in enumerate(individual_partition)
        treat_reg = []
        @suppress_err begin
            treat_reg = reg(panel_df[(panel_data["i"] .∈ Ref(j)) .* (panel_data["t"] .>= 1), :], @formula(Y ~  treat * t  + fe(i) + fe(t)), contrasts = Dict(:t => DummyCoding(base = Tstar-1), :treat => DummyCoding(base = 0)), save = true, Vcov.cluster(:i))
        end
        push!(treatment_temp, [vcat(treat_reg.coef[T+1:T+Tstar-2], [0] , treat_reg.coef[T+Tstar-1:end]), vcat(sqrt.(diag(treat_reg.vcov)[T+1:T+Tstar-2]), [Inf], sqrt.(diag(treat_reg.vcov)[T+Tstar-1:end]))])
        push!(temp_prob_num, mean((panel_data["i"] .∈ Ref(j)) .* (panel_data["treatment"] .== 1) ))
        push!(temp_treats, treat_reg)
    end
    push!(group_regs, temp_treats)
    treat_est[s] =  treatment_temp
    probs[s] = temp_prob_num ./ temp_prob_den
    match_main_permute
    println(match_main_permute)
    # Delta method with estimated probabilities
    panel_df = hcat(panel_df, select(panel_df, [:t => ByRow(isequal(v))=> Symbol(string("t=",v)) for v in Int.(unique(panel_df.t))]))
    for t in [1, 2, 3, 4, 5, 6]
        panel_df[:, string("t=",t,"treat")] = panel_df[:, string("t=",t)] .* panel_df[:, "treat"]
    end
    panel_df = hcat(panel_df, select(panel_df, [:j => ByRow(isequal(v))=> Symbol(string("j=",v)) for v in unique(range(1, length(individual_partition)))]))

    fe_df = temp_treats[1].fe
    fe_df[:, "resids"] = temp_treats[1].residuals
    for regj in temp_treats[2:end]
        temp_df = regj.fe
        temp_df[:, "resids"] = regj.residuals
        fe_df = vcat(fe_df, temp_df)
    end

    panel_df = leftjoin(panel_df, fe_df, on = [:i, :t])
    for j in range(1, length(match_main_permute))
        panel_df[:, string("j=",j,"treat")] = panel_df[:, string("j=",j)] .* panel_df[:, "treat"]
    end

    ztildes_errors = []
    ztildes = [] 
    i_order = []
    t_order = []
    indicator_join_list = []
    indicator_marg_list = []
    indicator_join_list_err = []
    indicator_marg_list_err = []
    for j in range(1, length(individual_partition))
        temp_panel = panel_df[panel_df.j .== j, :]
        partial_terms = partial_out(temp_panel, @formula(Y + treat * t ~  fe(i) + fe(t)), contrasts = Dict(:t => DummyCoding(base = Tstar-1), :treat => DummyCoding(base = 0)))[1]
        ytilde = partial_terms[:, 1]
        ztilde = partial_terms[:, end-4:end]
        push!(i_order, temp_panel.i)
        push!(t_order, temp_panel.t)
        push!(ztildes, Matrix(ztilde))
        temp_z_error = combine(groupby(hcat(temp_panel[:, ["i"]], ztilde .* temp_panel.resids), :i ), names(ztilde) .=> sum)
        push!(ztildes_errors, Matrix(temp_z_error[:, 2:end]) * sqrt( ((length(individual_partition[j]) * T)-1)/(length(individual_partition[j])*T -2*T )*(length(individual_partition[j])/(length(individual_partition[j])-1))))
        push!(indicator_join_list,  temp_panel[:, r"j=[0-9]treat$"])
        push!(indicator_marg_list, temp_panel.treat)
        
        push!(indicator_join_list_err, combine(groupby(temp_panel, :i),  names(temp_panel[:, r"j=[0-9]treat$"]) .=> mean)[:, 2:end] .- temp_prob_num')
        push!(indicator_marg_list_err, combine(groupby(temp_panel, :i),  :treat => mean)[:, 2:end] .- temp_prob_den)
    end

    ztilde_matrix = zeros(num_I*T, length(match_main_permute)*(T-1) + length(match_main_permute))
    ztilde_err_mat = zeros(num_I, length(match_main_permute)*(T-1) + length(match_main_permute))
    starting_row = 1
    starting_row_err = 1
    for (idx, j) in enumerate(range(1, length(match_main_permute)))
        nj = size(ztildes[j])[1]
        ztilde_matrix[starting_row:starting_row - 1 + nj, (idx-1)*(T-1)+1:(idx)*(T-1)] = ztildes[j]
        ztilde_matrix[starting_row:starting_row - 1 + nj, 3*(T-1)+1:3*(T-1)+3] = Matrix(indicator_join_list[j])
        #ztilde_matrix[starting_row:starting_row - 1 + nj, end] = indicator_marg_list[j]
        starting_row += nj

        nj = size(ztildes_errors[j])[1]
        ztilde_err_mat[starting_row_err:starting_row_err - 1 + nj, (idx-1)*(T-1)+1:(idx)*(T-1)] = ztildes_errors[j]
        ztilde_err_mat[starting_row_err:starting_row_err - 1 + nj, 3*(T-1)+1:3*(T-1)+3] = Matrix(indicator_join_list_err[j])
        #ztilde_err_mat[starting_row_err:starting_row_err - 1 + nj, end] = Matrix(indicator_marg_list_err[j])
        
        starting_row_err += nj
    end
    grad_array = []
    temp_index = []
    var_mat = ((inv(ztilde_matrix' * ztilde_matrix) * ztilde_err_mat') * (inv(ztilde_matrix' * ztilde_matrix) * ztilde_err_mat' )')

    temp_agg_ses = zeros(T-1)
        for t in range(1, T-1)
            r=t
            if t >= Tstar-1
                r+=1
            end
            grad_array =  temp_prob_num ./ temp_prob_den
            for j in range(1, length(match_main_permute))
                temp_grad = 0
                for j2 in setdiff(range(1, length(match_main_permute)), j)
                    temp_grad -= (temp_prob_num[j2] ./ (temp_prob_den ^ 2)) * treatment_temp[j2][1][r]
                end
                temp_grad +=  treatment_temp[j][1][r] * ((temp_prob_den - temp_prob_num[j])/(temp_prob_den ^ 2))
                push!(grad_array, temp_grad)
            end
            temp_index = vcat([t+ (x-1)*(T-1) for x in range(1,length(individual_partition))], (T-1)* length(individual_partition) .+ [x for x in range(1,length(individual_partition))])
            temp_var_mat = var_mat[temp_index, temp_index]
            temp_agg_ses[t]  = sqrt(grad_array' * temp_var_mat * grad_array)
        end
        insert!(temp_agg_ses, Tstar-1, Inf)
        treat_agg_ses[s] = temp_agg_ses

end

    
# Check Coverage Probabilities
trend_estimates = [convert_estim_to_mat(trend_est[s, 1]) for s in range(1, num_sims)] 
trend_ses = [convert_se_to_mat(trend_est[s, 1]) for s in range(1, num_sims)] 
treat_estimates = [convert_estim_to_mat(treat_est[s, 1]) for s in range(1, num_sims)] 
treat_ses = [convert_se_to_mat(treat_est[s, 1]) for s in range(1, num_sims)] 

group_order = [1, 2, 3]
treat_agg_est = [sum([x * y for (x,y) in zip(prob, eachrow(treat_estimate))]) for (prob, treat_estimate) in zip(probs, treat_estimates)]


#treat_agg_ses = [sqrt.(sum([x * y .^2 * x for (x,y) in zip(prob, eachrow(treat_se))])) for (prob, treat_se) in zip(probs, treat_ses)]

altprobs = [[(.8 *.203125)/.5, (.5 *.59375)/.5, (.2 *.203125)/.5][group_order] for i in range(1,num_sims)]
treat_alt_est = [sum([x * y for (x,y) in zip(prob, eachrow(treat_estimate))]) for (prob, treat_estimate) in zip(altprobs, treat_estimates)]
treat_alt_ses = [sqrt.(sum([x * y .^ 2 * x for (x,y) in zip(prob, eachrow(treat_se))])) for (prob, treat_se) in zip(altprobs, treat_ses)]







cp_stan_trend = zeros(size(trend_estimates[1]))
cp_stan_treat = zeros(size(treat_estimates[1]))
cp_stan_agg = zeros(size(treat_agg_est[1]))
cp_true_agg = zeros(size(treat_agg_est[1]))

true_trend = [params[1][j]  for j in group_order, t in range(2, Tstar-1)]
true_treat = [params[2][j] * (t >= Tstar)   for j in group_order, t in range(1, Tstar)]
true_treat = hcat(true_treat, 3 * true_treat[:,end])



α = .05 
z95 = quantile(Normal(0,1), (1 - (α/2)))
true_agg_treat = [0, 0, 0, 0, .638125, 1.914375]
for s in range(1, num_sims)
    cp_stan_trend += (trend_estimates[s] - trend_ses[s] * z95 .<= true_trend) .* (true_trend .<= trend_estimates[s] + trend_ses[s]* z95)
    cp_stan_treat += (treat_estimates[s] - treat_ses[s] * z95 .<= true_treat) .* (true_treat .<= treat_estimates[s] + treat_ses[s]* z95)
    cp_stan_agg += (treat_agg_est[s] - treat_agg_ses[s] * z95 .<= true_agg_treat) .* (true_agg_treat .<= treat_agg_est[s] + treat_agg_ses[s]* z95)
    cp_true_agg += (treat_alt_est[s] - treat_alt_ses[s] * z95 .<= true_agg_treat) .* (true_agg_treat .<= treat_alt_est[s] + treat_alt_ses[s]* z95)
end

println("Standard Trend: $(cp_stan_trend /  num_sims)")
println("Standard Treat: $(cp_stan_treat /  num_sims)")
println("Agg Treat: $(cp_stan_agg /  num_sims)")
println("Agg Treat: $(cp_true_agg /  num_sims)")

