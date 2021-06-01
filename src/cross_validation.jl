# Different cross-validation techniques
using Random;
# Holdout
function Holdout(ids :: Any, train_frac :: Float64, shuffle_id :: Bool = false)
    ids_mat = collect(ids)
    if shuffle_id
        Random.shuffle!(ids_mat)
    end
    break_point = trunc(Int64, round(size(ids_mat)[1] * train_frac))
    train_test_pairs = (ids_mat[1:break_point], ids_mat[(break_point + 1):size(ids_mat)[1]])
    return train_test_pairs
end
# KFold
function KFold(ids :: Any, k :: Int64, shuffle_id ::Bool = false)
    ids_mat = collect(ids)
    if shuffle_id
        Random.shuffle!(ids_mat)
    end
    paired_mat = [ids_mat vcat(repeat(collect(1:k), inner = div(size(ids_mat)[1], k)), collect(1:k)[1:rem(size(ids_mat)[1], k)])]
    train_test_pairs = []
    for i in 1:k
        train_test = (paired_mat[paired_mat[:, 2] .!= i, 1], paired_mat[paired_mat[:, 2] .== i, 1])
        push!(train_test_pairs, train_test)
    end
    return train_test_pairs
end
# Grouped KFold      
function GroupedKFold(ids :: Any, group_list :: Any)
    paired_mat = [collect(ids) collect(group_list)]
    train_test_pairs = []
    for groups in unique(group_list)
        train_test = (paired_mat[paired_mat[:, 2] .!= groups, 1], paired_mat[paired_mat[:, 2] .== groups, 1])
        push!(train_test_pairs, train_test)
    end
    return train_test_pairs
end
