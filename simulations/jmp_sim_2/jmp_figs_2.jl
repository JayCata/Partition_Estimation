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
# Set Working Dorectory and include ModularTrees.jl
cd(@__DIR__)
include("..\\..\\src\\part_est.jl")

function generate_trend!(panel_data, params, num_I, T, Tstar, Γ)
    eps_dis = Normal(0,.5)
    
    Yold = 1 .+ rand(num_I) + rand(eps_dis, num_I)
    Y = Yold
    Ydiff = fill(Inf, num_I)
    for t in range(2,T)
      
        Ynew = Yold + sum(params[1]'  .* panel_data["Γ"][(t-2) * num_I + 1:num_I*(t-1),:], dims = 2) + rand(eps_dis, num_I) .+ ((t >= Tstar) * (Γ * params[2] * (t - Tstar + 1)) .* (panel_data["treatment"][(t-2) * num_I + 1:num_I*(t-1),:]) )
        
        # Deal with group 1
        group_1_adjust = panel_data["Γ"][(t-2) * num_I + 1:num_I*(t-1), 1] .* panel_data["treatment"][(t-2) * num_I + 1:num_I*(t-1)] * -.2 + panel_data["Γ"][(t-2) * num_I + 1:num_I*(t-1) , 1] .* (1 .- panel_data["treatment"][(t-2) * num_I + 1:num_I*(t-1)]) * .8
        Ynew += group_1_adjust
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

function plot_partition(qgrid, f, grid_size)
    fig, ax = plt.subplots()
    function_vals = f.([qgrid[i, :] for i in range(1,size(qgrid)[1])])
    function_vals = [ifelse(x == 1, 1, ifelse(x == 3, 3, 2)) for x in function_vals]
    function_vals = reshape(function_vals, grid_size)
    #function_vals = replace(function_vals, 2 => 3, 3 => 2)
    ax.contourf([x  for x in unique(qgrid[:,1]), y in unique(qgrid[:,1]) ], 
                [y  for x in unique(qgrid[:,1]), y in unique(qgrid[:,1]) ], 
                function_vals, cmap=:viridis)
    plt.xticks([], [])
    plt.yticks([], [])
    return fig,ax
end
# Simulation Invariant Parameters
# fixed randomness
Random.seed!(123)
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
num_I = 8000


# Generate data & Trends
panel_data, Γ = generate_data(num_I, T, k, l)
generate_trend!(panel_data, params, num_I, T, Tstar, Γ)

# Set minimum number of observations in leaf
minobs = log(T * num_I)

folder = "figs"

# Grow Tree
pre_trend_ind = 1 .< panel_data["t"] .< Tstar
full_tree = MT.grow_tree(panel_data["X"][pre_trend_ind,2:(Tstar-1)]', panel_data["Ydiff"][pre_trend_ind,:], panel_data["Q_tilde"][pre_trend_ind, :]', feature_idx, minobs, MT.ols, MT.sse; early_stop = true)

# Join Tree
joined_trees, group_list, joined_θlist, joined_trees_index, joined_αs, joined_data_indices = MT.rj_full_tree(deepcopy(full_tree), panel_data["X"][pre_trend_ind,2:Tstar-1]', panel_data["Ydiff"][pre_trend_ind, :]; contiguous = true)

# Cross validate for main tree or match num regions to main tree

best_tree_idx = [MT.model_select(joined_trees, panel_data["Ydiff"][pre_trend_ind, :], panel_data["X"][pre_trend_ind, 2:(Tstar-1)]', panel_data["Q_tilde"][pre_trend_ind, :]')]


best_joined_tree = deepcopy(joined_trees[best_tree_idx])[1]
best_joined_theta = deepcopy(joined_θlist[best_tree_idx])[1]
best_joined_groups = deepcopy(group_list[best_tree_idx])[1]
threshold_obj_join, Q_index_join =  MT.get_thresholds(best_joined_tree, best_joined_groups)




idxs = []
regs = []
trend_regs = []

temp_prob_num = []
temp_prob_den = mean((panel_data[ "treatment"] .== 1))
### DiD
individual_partition, panel_df = dict_to_df(best_joined_groups, joined_data_indices, panel_data, Tstar)
   
match_main_permute = [1,2,3]     
trend_regs = []
trend_form = @formula(Y ~  treat * t + fe(i) + fe(t))
treatment_temp = []
for j in individual_partition
    trend_reg = reg(panel_df[(panel_data["i"] .∈ Ref(j)), :], trend_form, contrasts = Dict(:t => DummyCoding(base = Tstar-1)), Vcov.cluster(:i), save = true)
    push!(trend_regs, trend_reg)
    push!(temp_prob_num, mean((panel_data["i"] .∈ Ref(j)) .* (panel_data["treatment"] .== 1) ))
    push!(treatment_temp, [vcat(trend_reg.coef[T+1:T+Tstar-2], [0] , trend_reg.coef[T+Tstar-1:end]), vcat(sqrt.(diag(trend_reg.vcov)[T+1:T+Tstar-2]), [Inf], sqrt.(diag(trend_reg.vcov)[T+Tstar-1:end]))])
end

panel_df[:, "j"] = zeros(Int64,nrow(panel_df))
for (idx, j) in enumerate(individual_partition)
    panel_df[:, "j"] += Int.(idx .* (panel_df[:, "i"] .∈ Ref(j)))
end

# Delta method with estimated probabilities
panel_df = hcat(panel_df, select(panel_df, [:t => ByRow(isequal(v))=> Symbol(string("t=",v)) for v in Int.(unique(panel_df.t))]))
for t in [1, 2, 3, 4, 5, 6]
    panel_df[:, string("t=",t,"treat")] = panel_df[:, string("t=",t)] .* panel_df[:, "treat"]
end
panel_df = hcat(panel_df, select(panel_df, [:j => ByRow(isequal(v))=> Symbol(string("j=",v)) for v in unique(range(1, length(individual_partition)))]))

fe_df = trend_regs[1].fe
fe_df[:, "resids"] = trend_regs[1].residuals
for regj in trend_regs[2:end]
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
drop_idx = 1
sel_idx = setdiff([x for x in range(1, length(match_main_permute))], drop_idx)
    
temp_agg_ses = zeros(T-1)
temp_agg_est = zeros(T-1)
temp_prob_den_star = mean((panel_df.j .∈ Ref(sel_idx)) .&& (panel_df.treat .== 1))
for t in range(1, T-1)
    r=t
    if t >= Tstar-1
        r+=1
    end
    grad_array =  temp_prob_num[sel_idx] ./ temp_prob_den_star
    for j in sel_idx
        temp_agg_est[t] += treatment_temp[j][1][r]  * temp_prob_num[j] / temp_prob_den_star
        temp_grad = 0

        for j2 in setdiff(sel_idx, j)
            temp_grad -= (temp_prob_num[j2] ./ (temp_prob_den_star ^ 2)) * treatment_temp[j2][1][r]
        end
        temp_grad +=  treatment_temp[j][1][r] * ((temp_prob_den_star - temp_prob_num[j])/(temp_prob_den_star ^ 2))
        push!(grad_array, temp_grad)
    end
    temp_index = vcat([t+ (x-1)*(T-1) for x in sel_idx], (T-1)* length(individual_partition) .+ [x for x in sel_idx])
    temp_var_mat = var_mat[temp_index, temp_index]
    temp_agg_ses[t]  = sqrt(grad_array' * temp_var_mat * grad_array)
end
insert!(temp_agg_ses, Tstar-1, Inf)
insert!(temp_agg_est, Tstar-1, 0)


group_convert = [1, 2, 3]

tot_trend_reg = reg(panel_df, trend_form, contrasts = Dict(:t => DummyCoding(base = Tstar-1)), Vcov.robust())
β_naive = vcat(tot_trend_reg.coef[T+1:T+Tstar-2], [0] , tot_trend_reg.coef[(T+Tstar-1):end])
var_naive = diag(tot_trend_reg.vcov[T+1:end, T+1:end])
insert!(var_naive, Tstar-1, Inf)
 # Calculate weighted 
 probs = []
 β_j = []
 vars= []
ind_treat = combine(groupby(panel_df[:, ["i", "treat"]],  ["i"]), :treat => mean)
 for (j, idx) in enumerate(individual_partition)
     push!(β_j, vcat(trend_regs[j].coef[T+1:T+Tstar-2], [0] , trend_regs[j].coef[(T+Tstar-1):end]))
     prob_temp = mean(ind_treat.treat_mean[( ind_treat[!,"i"] .∈ Ref(idx)) ]) * ((length(idx)/num_I)/ mean(ind_treat.treat_mean))
     push!(probs, prob_temp)
     temp_var = diag(trend_regs[j].vcov[T+1:end, T+1:end])
     push!(vars, insert!(temp_var, Tstar-1, Inf))
 end
 β_j = vcat(β_j'...)
 β = (probs' * β_j)[1, :]
 Δvars = []

 for t in range(1, T)
     var_jt=  Array{Float64,1}()
     for var in vars
        if sum(isnan.(var)) == 0
            push!(var_jt, var[t])
        else
            push!(var_jt, 0)
        end
     end
     
     push!(Δvars, probs' * diagm(var_jt) * probs)
 end

# Plot
zscore = quantile(Normal(0,1), .975)
time_vec = vcat([x for x in range(1, T)])
group_cis = []
group_plot = plot(time_vec, zeros(length(time_vec)), linecolor=:black, alpha=.5, label = "", legend = :topleft, 
colorbar = true)
vline!([Tstar-1], linestyle=:dash, linecolor=:black, alpha=.5, label = "")
for (j, group_num) in enumerate(unique(best_joined_groups)[group_convert])
    ci_len = sqrt.(vars[group_num]) * zscore
    Plots.plot!(time_vec, β_j[group_num, :]; yerror =  ci_len, label = "Group $(j)", markerstrokecolor=:auto, linestyle = :dash, marker=(:circle,5), dpi=600) 
    push!(group_cis, ci_len)
end

ylabel!("βⱼₜ", fontsize = 40)
xlabel!("Period")
group_plot

plotpath = string(folder, "\\all_grp_2.png")
savefig(group_plot,  plotpath)
#Plots.plot!(time_vec, weighted_estimate, label = "Group Weight", markerstrokecolor=:auto, linestyle = :dash, marker=(:circle,5), alpha = .4, dpi=600)

β_j[1,1:Tstar-1] - group_cis[1][1:Tstar-1].<= 0  .<   β_j[1,1:Tstar-1] + group_cis[1][1:Tstar-1]

bad_groups = []
for j in range(1, length(individual_partition))
    if sum(β_j[j,1:Tstar-1] - group_cis[j][1:Tstar-1].<= 0  .<   β_j[j,1:Tstar-1] + group_cis[j][1:Tstar-1]) == length(β_j[j,1:Tstar-1])
        push!(bad_groups, false)
    else
        push!(bad_groups, true)
    end
end
empty_groups = probs .== 0
remove_groups = [1]
# Exclude Group
p_idx = setdiff(range(1,best_joined_tree.regions), remove_groups)
p_individual_partition = individual_partition[p_idx]
p_ind_rec =  unique(panel_df[panel_df.i .∈  Ref(vcat(p_individual_partition ...)), "i"] )

p_ninds = length(vcat(p_ind_rec...))
p_ind_treat = combine(groupby(panel_df[panel_df.i .∈ Ref(p_ind_rec), ["i", "treat"]],  ["i"]), :treat => mean)
 # Calculate weighted conditional
pprobs = []
pβ_j = []
pvars= []
for (j, idx) in zip(p_idx, p_individual_partition)
    push!(pβ_j, vcat(trend_regs[j].coef[T+1:T+Tstar-2], [0] , trend_regs[j].coef[(T+Tstar-1):end]))
    prob_temp = mean(ind_treat.treat_mean[( ind_treat[!,"i"] .∈ Ref(idx)) ]) * ((length(idx)/num_I)/ mean(ind_treat.treat_mean))
    push!(pprobs, prob_temp)
    temp_var = diag(trend_regs[j].vcov[T+1:end, T+1:end])
    push!(pvars, insert!(temp_var, Tstar-1, Inf))
end
 pβ_j = vcat(pβ_j'...)
 pprobs = [(.5 *.59375)/ ((.5 *.59375 + .2 *.203125)), (.2 *.203125)/ ((.5 *.59375 + .2 *.203125))]

 pβ = (pprobs' * pβ_j)[1, :]
 pΔvars = []

for t in range(1, T-1)
    var_jt=  Array{Float64,1}()
    for var in pvars
       if sum(isnan.(var)) == 0
           push!(var_jt, var[t])
       else
           push!(var_jt, 0)
       end
    end
    
    push!(pΔvars, pprobs' * diagm(var_jt) * pprobs)
end


 # group plot for select regions
group_plot = plot(time_vec, zeros(length(time_vec)), linecolor=:black, alpha=.5, label = "", legend = :topleft)
vline!([Tstar-1], linestyle=:dash, linecolor=:black, alpha=.5, label = "")
for (j, group_num) in enumerate(unique(best_joined_groups)[group_convert])
    if group_num ∉ remove_groups
        ci_len = sqrt.(vars[group_num]) * zscore
        Plots.plot!(time_vec, β_j[group_num, :]; yerror =  ci_len, label = "Group $(j)", markerstrokecolor=:auto, alpha = .3, linestyle = :dash, marker=(:circle,5), dpi=600) 
        push!(group_cis, ci_len)
    end
end



ylabel!("βⱼₜ", fontsize = 40)
xlabel!("Period")



naive_ci_len = zscore * sqrt.(var_naive)
Plots.plot!(time_vec, β_naive; yerror =  naive_ci_len, label = "Naive TWFE", markerstrokecolor=:auto, linestyle = :dash, marker=(:circle,5), alpha = 1, dpi=600) 

weight_ci_len = zscore * temp_agg_ses
Plots.plot!(time_vec, temp_agg_est; yerror =  weight_ci_len, label = "Partial Group Aggregate", markerstrokecolor=:auto, linestyle = :dash, marker=(:circle,5), alpha = 1, dpi=600) 

plotpath = string(folder, "\\sel_grp_2.png")
group_plot
savefig(group_plot,  plotpath)


# Generate partition
using PyPlot
joined_tree_group(q) = MT.data_groups(best_joined_tree, q, best_joined_groups)[1]
grid_size_1 = 1000
grid_size_2 = 1000
grid_range_1 = range(-2, 2, grid_size_1)
grid_range_2 = range(-2, 2, grid_size_2)
qgrid = hcat([[i,j] for i in grid_range_1, j in grid_range_1]...)'
comp_plt = plot_partition(qgrid, joined_tree_group, (grid_size_1, grid_size_2))
comp_plt
xlabel("Q¹", fontsize = 20)
ylabel("Q²",  fontsize = 20)
plt.annotate(L"\hat \Gamma_1", (-1.5, -1.5), fontsize = 28, color = "white", horizontalalignment= "center", verticalalignment = "center")
plt.annotate(L"\hat \Gamma_2", (0, 0), fontsize = 28, color = "white", horizontalalignment= "center", verticalalignment = "center")
plt.annotate(L"\hat \Gamma_3", (1.5, 1.5), fontsize = 28, color = "black", horizontalalignment= "center", verticalalignment = "center")


display(gcf())
plt.savefig(string(folder,"\\est_partition_2.png"))

