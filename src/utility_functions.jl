# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
using Statistics
using Distributed

# Loss function for Flux models
# function loss(flux_model, loss_init, x, y)
#     return loss_init(flux_model(x), y)
# end
function loss(flux_model, loss_init, x, y)
    return sum(loss_init(flux_model(xi), yi) for (xi, yi) in zip(x, y))
end	
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, loss_init, data, optimizer)
    for d in data
        gs = Flux.gradient(Flux.params(flux_model)) do
            train_loss = loss(flux_model, loss_init, d...)
        end
        Flux.update!(optimizer, Flux.params(flux_model), gs)
    end
end
# Scaler for data normalization and standardization
# Construct maximum minimum scaler for data normalization
mutable struct max_min_scaler
    min :: Union{Float64, Int64}
    max :: Union{Float64, Int64}
end
# Construct standar scaler for data standardization
mutable struct standard_scaler
end
# Function for fitting the scaler
function fit_scaler(scaler :: Union{max_min_scaler, standard_scaler}, raw_x :: Union{Matrix{Float64}, Matrix{Int64}})
	if typeof(raw_x) == Matrix{Int64}
		raw_x = convert.(Float64, raw_x)
	end
    if typeof(scaler) == max_min_scaler
		max_raw_x, min_raw_x = [], []
		for j in Distributed.splitrange(1, size(raw_x)[2], Threads.nthreads())
			t = Threads.@spawn begin
				for i in j
					max_raw_x0 = extrema(raw_x[:, i])[2]
					min_raw_x0 = extrema(raw_x[:, i])[1]
					if max_raw_x0 - min_raw_x0 == 0.0
						max_raw_x1 = 1.0
						min_raw_x1 = 0.0
					else
						max_raw_x1 = max_raw_x0
						min_raw_x1 = min_raw_x0
					end
					push!(min_raw_x, min_raw_x1)
					push!(max_raw_x, max_raw_x1)
				end
			end
			fetch(t)
		end
        return Dict(:max => scaler.max, :min => scaler.min, :max_value => max_raw_x, :min_value => min_raw_x)
    elseif typeof(scaler) == standard_scaler
        mean_raw_x, stdev_raw_x = [], []
		for j in Distributed.splitrange(1, size(raw_x)[2], Threads.nthreads())
			t = Threads.@spawn begin
				for i in j
					mean_raw_x0 = mean(raw_x[:, i])
					stdev_raw_x0 = sqrt(var(raw_x[:, i]))
					if stdev_raw_x0 == 0.0
						stdev_raw_x1 = 1.0
						mean_raw_x1 = 0.0
					else
						stdev_raw_x1 = stdev_raw_x0
						mean_raw_x1 = mean_raw_x0
					end
		            push!(mean_raw_x, mean_raw_x0)
		            push!(stdev_raw_x, stdev_raw_x1)
				end
			end
			fetch(t)
        end
        return Dict(:mean_value => mean_raw_x, :stdev_value => stdev_raw_x)
	else
        return nothing
    end
end
# Function for transforming a dataset by using fitted scaler parameters
function scale_transform(fitted_scaler :: Dict, raw_x :: Union{Matrix{Float64}, Matrix{Int64}})
	if typeof(raw_x) == Matrix{Int64}
		raw_x = convert.(Float64, raw_x)
	end
    conv_x = Matrix{Float64}(undef, size(raw_x)[1], size(raw_x)[2])
	for j in Distributed.splitrange(1, size(raw_x)[2], Threads.nthreads())
		t = Threads.@spawn begin
			for i in j
		        if (haskey(fitted_scaler, :max_value) == true) & (haskey(fitted_scaler, :min_value) == true) & (haskey(fitted_scaler, :max) == true) & (haskey(fitted_scaler, :min) == true)
					conv_x[:, i] .= fitted_scaler[:min] .+ (raw_x[:, i] .- fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) .* (fitted_scaler[:max] - fitted_scaler[:min])
		        elseif (haskey(fitted_scaler, :mean_value) == true) & (haskey(fitted_scaler, :stdev_value) == true)
					conv_x[:, i] .= (raw_x[:, i] .- fitted_scaler[:mean_value][i]) ./ fitted_scaler[:stdev_value][i]
				else
		            conv_x[:, i] .= raw_x[:, i]
		        end
			end
		end
		fetch(t)
    end
    return conv_x
end
# Function for converting the transformed data back to original data by using fitted scaler parameters
function scale_back(fitted_scaler :: Dict, conv_x :: Union{Matrix{Float64}, Matrix{Int64}})
	if typeof(conv_x) == Matrix{Int64}
		conv_x = convert.(Float64, conv_x)
	end
	raw_x = Matrix{Float64}(undef, size(conv_x)[1], size(conv_x)[2])
    for j in Distributed.splitrange(1, size(conv_x)[2], Threads.nthreads())
		t = Threads.@spawn begin
			for i in j
		        if (haskey(fitted_scaler, :max_value) == true) & (haskey(fitted_scaler, :min_value) == true) & (haskey(fitted_scaler, :max) == true) & (haskey(fitted_scaler, :min) == true)
		            raw_x[:, i] .= fitted_scaler[:min_value][i] .+ (conv_x[:, i] .- fitted_scaler[:min]) .* (fitted_scaler[:max_value][i] - fitted_scaler[:min_value][i]) ./ (fitted_scaler[:max] - fitted_scaler[:min])
				elseif (haskey(fitted_scaler, :mean_value) == true) & (haskey(fitted_scaler, :stdev_value) == true)
					raw_x[:, i] .= fitted_scaler[:mean_value][i] .+ conv_x[:, i] .* fitted_scaler[:stdev_value][i]
				else
		            raw_x[:, i] .= conv_x[:, i]
		        end
			end
		end
		fetch(t)
   end
   return raw_x
end
