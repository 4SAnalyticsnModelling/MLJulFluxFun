# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
# Loss function for Flux models (with L2 reuglarization)
mutable struct lambda2
    l2_val :: Float64
end
function loss(flux_model, loss_init, x, y :: Vector, l2)
    y_pred = vec(flux_model(x)[1, :])
    sqnorm(x) = sum(abs2, x)
    return (loss_init(y_pred, y) + l2.l2_val * sum(sqnorm, Flux.params(flux_model)))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, loss_init, data, optimizer, l2)
    ps = Flux.params(flux_model)
    for d in data
        gs = Flux.gradient(ps) do
            train_loss = loss(flux_model, loss_init, d..., l2)
        end
        Flux.update!(optimizer, ps, gs)
    end
end
# Scaler for data normalization and standardization
# Construct maximum minimum scaler for data normalization
mutable struct max_min_scaler
    min :: Float64
    max :: Float64
end
# Construct standar scaler for data standardization
mutable struct standard_scaler
end
# Function for fitting the scaler
function fit_scaler(scaler :: Union{max_min_scaler, standard_scaler}, raw_x :: Matrix{Float64})
    if typeof(scaler) == max_min_scaler
        i = 1
        max_raw_x, min_raw_x = [], []
        while i < (size(raw_x)[2] + 1)
            push!(max_raw_x, extrema(raw_x[:, i])[2])
            push!(min_raw_x, extrema(raw_x[:, i])[1])
            i += 1
        end
        return Dict(:max => scaler.max, :min => scaler.min, :max_value => max_raw_x, :min_value => min_raw_x)
    elseif typeof(scaler) == standard_scaler
        i = 1
        mean_raw_x, stdev_raw_x = [], []
        while i < (size(raw_x)[2] + 1)
            push!(mean_raw_x, sum(raw_x[:, i]) / size(raw_x)[1])
            push!(stdev_raw_x, sqrt(sum((raw_x[:, i] .- sum(raw_x[:, i]) / size(raw_x)[1]).^2.0) / (size(raw_x)[1] - 1)))
            i += 1
        end
        return Dict(:mean_value => mean_raw_x, :stdev_value => stdev_raw_x)
    else
        return nothing
    end
end
# Function for transforming a dataset by using fitted scaler parameters
function scale_transform(fitted_scaler :: Dict, raw_x :: Matrix{Float64})
    i = 1
    conv_x = Matrix{Float64}(undef, size(raw_x)[1], size(raw_x)[2])
    while i < (size(raw_x)[2] + 1)
        if length(fitted_scaler) == 4
            if fitted_scaler[:max_value][i] > fitted_scaler[:min_value][i]
                conv_x[:, i] .= fitted_scaler[:min] .+ (raw_x[:, i] .- fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) .* (fitted_scaler[:max] - fitted_scaler[:min])
            else
                conv_x[:, i] .= raw_x[:, i]
            end
        elseif length(fitted_scaler) == 2
            if fitted_scaler[:stdev_value][i] > 0
                conv_x[:, i] .= (raw_x[:, i] .- fitted_scaler[:mean_value][i]) ./ fitted_scaler[:stdev_value][i]
            else
                conv_x[:, i] .= raw_x[:, i]
            end
        else
            conv_x[:, i] .= raw_x[:, i]
        end
        i += 1
    end
    return conv_x
end
# Function for converting the transformed data back to original data by using fitted scaler parameters
function scale_back(fitted_scaler :: Dict, conv_x :: Matrix{Float64})
    i = 1
    raw_x = Matrix{Float64}(undef, size(conv_x)[1], size(conv_x)[2])
    while i < (size(raw_x)[2] + 1)
        if length(fitted_scaler) == 4
            if fitted_scaler[:max_value][i] > fitted_scaler[:min_value][i]
                raw_x[:, i] .= fitted_scaler[:min_value][i] .+ (conv_x[:, i] .- fitted_scaler[:min]) .* (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max] - fitted_scaler[:min])
            else
                raw_x[:, i] .= conv_x[:, i]
            end
        elseif length(fitted_scaler) == 2
            if fitted_scaler[:stdev_value][i] > 0
                raw_x[:, i] .= fitted_scaler[:mean_value][i] .+ (conv_x[:, i] .* fitted_scaler[:stdev_value][i])
            else
                raw_x[:, i] .= conv_x[:, i]
            end
        else
            raw_x[:, i] .= conv_x[:, i]
        end
        i += 1
   end
   return raw_x
end
