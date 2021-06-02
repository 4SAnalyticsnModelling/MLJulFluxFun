# Different cross-validation techniques
using Random;
# Holdout
function Holdout(ids :: T, train_frac :: Float64, shuffle_id :: Bool = false, nsample :: Int64 = 1) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}}}
    train_test_pairs = []
    ids_mat = collect(ids)
    for i in 1:nsample
        if shuffle_id
            Random.shuffle!(ids_mat)
        end
        break_point = trunc(Int64, round(size(ids_mat)[1] * train_frac))
        train_test = (ids_mat[1:break_point], ids_mat[(break_point + 1):size(ids_mat)[1]])
        train_test_pairs = vcat(train_test_pairs, train_test)
    end
    return train_test_pairs
end
# KFold
function KFold(ids :: T, k :: Int64, shuffle_id ::Bool = false, nsample :: Int64 = 1) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}}}
    ids_mat = collect(ids)
    train_test_pairs = []
    for i in 1:nsample
        if shuffle_id
            Random.shuffle!(ids_mat)
        end
        paired_mat = [ids_mat vcat(repeat(collect(1:k), inner = div(size(ids_mat)[1], k)), collect(1:k)[1:rem(size(ids_mat)[1], k)])]
        for i in 1:k
            train_test = (paired_mat[paired_mat[:, 2] .!= i, 1], paired_mat[paired_mat[:, 2] .== i, 1])
            push!(train_test_pairs, train_test)
        end
    end
    return train_test_pairs
end
# Grouped KFold
function GroupedKFold(ids :: T, group_list :: S) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}}, S <: Union{Vector{String}, Vector{Int64}}}
    paired_mat = [collect(ids) collect(group_list)]
    train_test_pairs = []
    for groups in unique(group_list)
        train_test = (convert.(Int64, paired_mat[paired_mat[:, 2] .!= groups, 1]), convert.(Int64, paired_mat[paired_mat[:, 2] .== groups, 1]))
        train_test_pairs = vcat(train_test_pairs, train_test)
    end
    return train_test_pairs
end
