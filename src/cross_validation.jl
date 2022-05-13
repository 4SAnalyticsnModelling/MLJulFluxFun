# Different cross-validation techniques
using Random
using Distributed
using DataFrames
# Construct cross-validation strategies
# Holdout_
mutable struct Holdout_
    train_frac :: Float64
    shuffle_id :: Bool
    nsample :: Int64
end
# KFold_
mutable struct KFold_
    k :: Int64
    shuffle_id :: Bool
    nsample :: Int64
end
# Grouped KFold_
mutable struct GroupedKFold_
    group_list :: Union{Vector{String}, Vector{Int64}}
end

# Cross-validation function
function cross_validate(cv_name :: Union{Holdout_, KFold_, GroupedKFold_}, ids :: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}})
    if typeof(cv_name) == Holdout_
        train_test_pairs = []
        ids_mat = collect(ids)
        for m in Distributed.splitrange(1, cv_name.nsample, Threads.nthreads())
            t = Threads.@spawn begin
                for i in m
                    if cv_name.shuffle_id
                        Random.shuffle!(ids_mat)
                    end
                    break_point = trunc(Int64, round(size(ids_mat)[1] * cv_name.train_frac))
                    train_test = (ids_mat[1:break_point], ids_mat[(break_point + 1):size(ids_mat)[1]])
                    push!(train_test_pairs, train_test)
                end
            end
            fetch(t)
        end
        return train_test_pairs
    elseif typeof(cv_name) == KFold_
        train_test_pairs = []
        ids_mat = collect(ids)
        for m in Distributed.splitrange(1, cv_name.nsample, Threads.nthreads())
            t = Threads.@spawn begin
                for i in m
                    if cv_name.shuffle_id
                        Random.shuffle!(ids_mat)
                    end
                    for k in Distributed.splitrange(1, length(ids_mat), cv_name.k)
                        train_test = (ids_mat[Not(k)], ids_mat[k])
                        push!(train_test_pairs, train_test)
                    end
                end
            end
            fetch(t)
        end
        return train_test_pairs
    elseif typeof(cv_name) == GroupedKFold_
        train_test_pairs = []
        ids_mat = collect(ids)
        # paired_mat = [collect(ids) collect(cv_name.group_list)]
        for groups in unique(cv_name.group_list)
            k = findall(q -> q == groups, cv_name.group_list)
            train_test = (ids_mat[Not(k)], ids_mat[k])
            push!(train_test_pairs, train_test)
        end
        return train_test_pairs
    else
        return nothing
    end
end
