# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
using Statistics
# Loss function for Flux models (with L2 reuglarization)
mutable struct lambda2
    l2_val :: Float64
end
function loss(flux_model, loss_init, x, y, l2)
    sqnorm(x) = sum(abs2, x)
    return (loss_init(flux_model(x), y) + l2.l2_val * sum(sqnorm, Flux.params(flux_model)))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, loss_init, data, optimizer, l2)
    for d in data
        gs = Flux.gradient(Flux.params(flux_model)) do
            train_loss = loss(flux_model, loss_init, d..., l2)
        end
        Flux.update!(optimizer, Flux.params(flux_model), gs)
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
# Construct robust scaler for data standardization (for dataset with outliers)
mutable struct robust_scaler
end
# Function for fitting the scaler
function fit_scaler(scaler :: Union{max_min_scaler, standard_scaler, robust_scaler}, raw_x :: Matrix{Float64})
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
            push!(mean_raw_x, mean(raw_x[:, i]))
            push!(stdev_raw_x, sqrt(var(raw_x[:, i])))
            i += 1
        end
        return Dict(:mean_value => mean_raw_x, :stdev_value => stdev_raw_x)
	elseif typeof(scaler) == robust_scaler
		i = 1
		median_raw_x, p75_raw_x, p25_raw_x = [], [], []
		while i < (size(raw_x)[2] + 1)
			push!(median_raw_x, median(raw_x[:, i]))
			push!(p75_raw_x, quantile(raw_x[:, i], 0.75))
			push!(p25_raw_x, quantile(raw_x[:, i], 0.25))
			i += 1
		end
		return Dict(:median_value => median_raw_x, :p75_value => p75_raw_x, :p25_value => p25_raw_x)
    else
        return nothing
    end
end
# Function for transforming a dataset by using fitted scaler parameters
function scale_transform(fitted_scaler :: Dict, raw_x :: Matrix{Float64})
    i = 1
    conv_x = Matrix{Float64}(undef, size(raw_x)[1], size(raw_x)[2])
    while i < (size(raw_x)[2] + 1)
        if (haskey(fitted_scaler, :max_value) == true) & (haskey(fitted_scaler, :min_value) == true) & (haskey(fitted_scaler, :max) == true) & (haskey(fitted_scaler, :min) == true)
			if (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) > 1.0
		        conv_x[:, i] .= fitted_scaler[:min] .+ (raw_x[:, i] .- fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) .* (fitted_scaler[:max] - fitted_scaler[:min])
			else
				conv_x[:, i] .= raw_x[:, i]
	        end
        elseif (haskey(fitted_scaler, :mean_value) == true) & (haskey(fitted_scaler, :stdev_value) == true)
			if fitted_scaler[:stdev_value][i] > 1.0
            	conv_x[:, i] .= (raw_x[:, i] .- fitted_scaler[:mean_value][i]) ./ fitted_scaler[:stdev_value][i]
			else
				conv_x[:, i] .= raw_x[:, i]
	        end
		elseif (haskey(fitted_scaler, :median_value) == true) & (haskey(fitted_scaler, :p75_value) == true) & (haskey(fitted_scaler, :p25_value) == true)
			if (fitted_scaler[:p75_value][i] - fitted_scaler[:p25_value][i]) > 1.0
				conv_x[:, i] .= (raw_x[:, i] .- fitted_scaler[:median_value][i]) ./ (fitted_scaler[:p75_value][i] - fitted_scaler[:p25_value][i])
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
        if (haskey(fitted_scaler, :max_value) == true) & (haskey(fitted_scaler, :min_value) == true) & (haskey(fitted_scaler, :max) == true) & (haskey(fitted_scaler, :min) == true)
			if (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) > 1.0
	            raw_x[:, i] .= fitted_scaler[:min_value][i] .+ (conv_x[:, i] .- fitted_scaler[:min]) .* (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max] - fitted_scaler[:min])
			else
	            raw_x[:, i] .= conv_x[:, i]
	        end
		elseif (haskey(fitted_scaler, :mean_value) == true) & (haskey(fitted_scaler, :stdev_value) == true)
			if fitted_scaler[:stdev_value][i] > 1.0
				raw_x[:, i] .= fitted_scaler[:mean_value][i] .+ conv_x[:, i] .* fitted_scaler[:stdev_value][i]
			else
				raw_x[:, i] .= conv_x[:, i]
			end
		elseif (haskey(fitted_scaler, :median_value) == true) & (haskey(fitted_scaler, :p75_value) == true) & (haskey(fitted_scaler, :p25_value) == true)
			if (fitted_scaler[:p75_value][i] - fitted_scaler[:p25_value][i]) > 1.0
				raw_x[:, i] .= fitted_scaler[:median_value][i] .+ conv_x[:, i] .* (fitted_scaler[:p75_value][i] - fitted_scaler[:p25_value][i])
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
