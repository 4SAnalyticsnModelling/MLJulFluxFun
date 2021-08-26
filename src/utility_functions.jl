# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
# Root mean squared error
function rmse_(obs :: T, pred :: S) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}, S <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}}
    obs = convert.(Float64, collect(obs))
    pred = convert.(Float64, collect(pred))
    return sqrt(sum((obs .- pred).^2)/length(obs))
end
# Willmott's index of model agreement (d)
function willmott_d(obs :: T, pred :: S) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}, S <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}}
    obs = convert.(Float64, collect(obs))
    pred = convert.(Float64, collect(pred))
    return 1.0 - sum((obs .- pred).^2) / sum((abs.(pred .- sum(obs)/length(obs)) .+ abs.(obs .- sum(obs)/length(obs))).^2)
end
# Loss function for Flux models
function loss(flux_model, loss_init, x, y)
    y_pred = vec(flux_model(x))
    return loss_init(y_pred, vec(y))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, loss_init, data, optimizer)
    ps = Flux.params(flux_model)
    for d in data
        train_loss, back = Zygote.pullback(() -> loss(flux_model, loss_init, d...), ps)
        gs = back(one(train_loss))
        Flux.update!(optimizer, ps, gs)
    end
 end
