# Different cross-validation techniques
using Random;
# Construct cross-validation strategies
# Holdout
mutable struct Holdout
    train_frac :: Float64
    shuffle_id :: Bool
    nsample :: Int64
end
# KFold
mutable struct KFold
    k :: Int64
    shuffle_id :: Bool
    nsample :: Int64
end
# Grouped KFold
mutable struct GroupedKFold
    group_list :: Union{Vector{String}, Vector{Int64}}
end

# Cross-validation function
function cross_validate(cv_name :: Union{Holdout, KFold, GroupedKFold}, ids :: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}})
    train_test_pairs = []
    if typeof(cv_name) == Holdout
        ids_mat = collect(ids)
        for i in 1:cv_name.nsample
            if cv_name.shuffle_id
                Random.shuffle!(ids_mat)
            end
            break_poInt64 = trunc(Int64, round(size(ids_mat)[1] * cv_name.train_frac))
            train_test = (ids_mat[1:break_poInt64], ids_mat[(break_poInt64 + 1):size(ids_mat)[1]])
            train_test_pairs = vcat(train_test_pairs, train_test)
        end
        return train_test_pairs
    elseif typeof(cv_name) == KFold
        ids_mat = collect(ids)
        for i in 1:cv_name.nsample
            for i in 1:cv_name.k
                if cv_name.shuffle_id
                    Random.shuffle!(ids_mat)
                end
                paired_mat = [ids_mat vcat(repeat(collect(1:cv_name.k), inner = div(size(ids_mat)[1], cv_name.k)), collect(1:cv_name.k)[1:rem(size(ids_mat)[1], cv_name.k)])]
                train_test = (paired_mat[paired_mat[:, 2] .!= i, 1], paired_mat[paired_mat[:, 2] .== i, 1])
                push!(train_test_pairs, train_test)
            end
        end
        return train_test_pairs
    elseif typeof(cv_name) == GroupedKFold
        paired_mat = [collect(ids) collect(cv_name.group_list)]
        for groups in unique(cv_name.group_list)
            train_test = (convert.(Int64, paired_mat[paired_mat[:, 2] .!= groups, 1]), convert.(Int64, paired_mat[paired_mat[:, 2] .== groups, 1]))
            train_test_pairs = vcat(train_test_pairs, train_test)
        end
        return train_test_pairs
    else
        return nothing
    end
end
