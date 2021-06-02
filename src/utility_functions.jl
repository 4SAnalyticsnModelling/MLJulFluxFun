# Utility and wrapper functions to be used in training and evaluating models
using Flux;
# Root mean squared error
function rmse(obs :: T, pred :: S) where {T <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}, S <: Union{UnitRange{Int64}, StepRange{Int64, Int64}, Base.OneTo{Int64}, Vector{Int64}, UnitRange{Float64}, StepRange{Float64, Float64}, Vector{Float64}, Vector{Float32}, Vector{Number}}}
    return sqrt(sum((collect(obs) .- collect(pred)).^2)/length(collect(obs)))
end

# Loss function for Flux models
function loss(flux_model, x, y)
    return sqrt(sum((vec(y) .- vec(flux_model(x))).^2)/length(vec(y)))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, data, optimizer)
    ps = Flux.params(flux_model)
    for d in data
      gs = Flux.gradient(ps) do
        training_loss = loss(flux_model, d...)
      end
      Flux.update!(optimizer, ps, gs)
    end
 end
