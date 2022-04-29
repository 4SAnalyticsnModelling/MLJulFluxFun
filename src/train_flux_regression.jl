using Flux
using CSV
using DataFrames
using Statistics
using BSON
using Flux.Zygote
# This function evaluates flux models based on user defined resampling strategies
# Example of flux_model_builder
# mutable struct nnet_mod_builder
#     n1 :: Int
#     n2 :: Int
#     n3 :: Int
#     n4 :: Int
# end
# function nnet_build(nn :: nnet_mod_builder, n_in, n_out)
#     return Flux.Chain(Dense(n_in, nn.n1, relu, init = Flux.kaiming_normal),
#                  Dense(nn.n1, nn.n2, relu, init = Flux.kaiming_normal),
#                  Dense(nn.n2, nn.n3, relu, init = Flux.kaiming_normal),
#                  Dense(nn.n3, nn.n4, relu, init = Flux.kaiming_normal),
#                  Dense(nn.n4, n_out, init = Flux.kaiming_normal))
# end
function flux_mod_eval(flux_model_builder :: Any,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    cv_strategy :: Union{Holdout, KFold, GroupedKFold},
	cv_strategy_train :: Union{Holdout, KFold, GroupedKFold},
	scaler_x :: Union{Nothing, max_min_scaler, standard_scaler},
    n_epochs :: Int64,
    pullback :: Bool,
	lcheck :: Int64,
	nobs_per_batch :: Int64,
	save_trained_model :: Bool,
    loss_init,
    optimizer,
	r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    if save_trained_model
		mkdir(save_trained_model_at * "/saved_trained_model(s)")
		if isnothing(scaler_x) == false
		    mkdir(save_trained_model_at * "/saved_trained_Xscaler(s)")
	    end
    end
    model_perform = Array{Float64}(undef, 0, 8)
	cv_all = cross_validate(cv_strategy, eachindex(y))
    k = 1
    while k < (1 + size(cv_all)[1])
        train0, test = cv_all[k, ]
		cv_train = cross_validate(cv_strategy, train0)
		l = 1
		while l < (1 + size(cv_train)[1])
			flux_model = flux_model_builder
			train, valid = cv_train[l, ]
			if pullback == true
	            valid_loss_record = []
	            early_stop_flag = 0
	            model_dict = Dict()
	        end
			if isnothing(scaler_x) == false
	            x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
				if save_trained_model
	            	BSON.@save(save_trained_model_at * "/saved_trained_Xscaler(s)/Xscaler_" * string(k) * string(l) * ".bson", x_scaler)
				end
	            x_train = Matrix(scale_transform(x_scaler, Matrix(x[train, :]))')
				x_valid = Matrix(scale_transform(x_scaler, Matrix(x[valid, :]))')
	            x_test = Matrix(scale_transform(x_scaler, Matrix(x[test, :]))')
	        else
	            x_train = Matrix(Matrix(x[train, :])')
				x_valid = Matrix(Matrix(x[valid, :])')
	            x_test = Matrix(Matrix(x[test, :])')
	        end
	        y_train = Matrix(y[train]')
			y_valid = Matrix(y[valid]')
	        y_test = Matrix(y[test]')
        	data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
        	j = 1
	        while j < (n_epochs + 1)
	            my_custom_train!(flux_model, loss, loss_init, data, optimizer)
	            valid_loss = loss(flux_model, loss_init, x_valid, y_valid)
	            push!(valid_loss_record, valid_loss)
	            push!(model_dict, Symbol("model" * string(j)) => flux_model)
	            if isnan(valid_loss) == false
	                println("epoch = " * string(j) * " validation_loss = " * string(valid_loss))
	                if pullback == true
	                    if j > 1
	                        if valid_loss .>= valid_loss_record[j - 1]
	                            early_stop_flag += 1
	                        else
	                            early_stop_flag -= 1
	                        end
	                        early_stop_flag = max(0, min(lcheck, early_stop_flag))
	                        if early_stop_flag == lcheck
	                            try
	                                Flux.stop()
	                            catch
	                            finally
	                            end
	                            break
	                        end
	                    end
	                end
	            else
	                try
	                    Flux.skip()
	                catch
	                finally
	                end
	            end
	            j += 1
	        end
	        if early_stop_flag == lcheck
	            model = model_dict[Symbol("model" * string(j - early_stop_flag))]
	        else
	            model = model_dict[Symbol("model" * string(n_epochs))]
	        end
	        flux_model_pred = flux_model_builder
	        Flux.loadmodel!(flux_model_pred, model);
	        y_pred_test = flux_model_pred(x_test)
			y_pred_valid = flux_model_pred(x_valid)
	        y_pred_train = flux_model_pred(x_train)
	        r2_test = round((Statistics.cor(y_test[1, :], y_pred_test[1, :]))^2, digits = r_squared_precision)
	        rmse_test = round(sqrt(Flux.Losses.mse(y_pred_test, y_test)), digits = rmse_precision)
			r2_valid = round((Statistics.cor(y_valid[1, :], y_pred_valid[1, :]))^2, digits = r_squared_precision)
	        rmse_valid = round(sqrt(Flux.Losses.mse(y_pred_valid, y_valid)), digits = rmse_precision)
	        r2_train = round((Statistics.cor(y_train[1, :], y_pred_train[1, :]))^2, digits = r_squared_precision)
	        rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
	        model_perform = [k l r2_test r2_train r2_valid rmse_test rmse_valid rmse_train]
	        if (k == 1) & (l == 1)
	            CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:test_iter, :train_iter, :r_squared_test, :r_squared_train, :r_squared_valid, :rmse_test, :rmse_valid, :rmse_train]))
	        else
	            CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:test_iter, :train_iter, :r_squared_test, :r_squared_train, :r_squared_valid, :rmse_test, :rmse_valid, :rmse_train]), append = true)
	        end
			if save_trained_model
				BSON.@save(save_trained_model_at * "/saved_trained_model(s)/trained_model_" * string(k) * string(l) * ".bson", flux_model)
			end
			l += 1
	    end
		k += 1
	end
end

# This function chooses the final predictive model by using stacking ensemble method and training a meta-learner
function flux_mod_stack(flux_model_builder :: Any,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    scaler_x :: Union{Nothing, max_min_scaler, standard_scaler},
    n_epochs :: Int64,
    nobs_per_batch :: Int64,
	save_trained_model :: Bool,
    loss_init,
    optimizer,
	n_boot :: Int64,
	r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    if save_trained_model
		mkdir(save_trained_model_at * "/saved_ensembled_meta_model(s)")
		if isnothing(scaler_x) == false
		    mkdir(save_trained_model_at * "/saved_ensembled_meta_Xscaler(s)")
	    end
    end
    model_perform = Array{Float64}(undef, 0, 3)
	x_meta = DataFrame()
	y_meta = []
	k = 1
    while k < (1 + n_boot)
    	flux_model = flux_model_builder
		# Perform bootstrap (bagging) training
		train = [rand(eachindex(y)) for _ in 1:length(y)]
		append!(x_meta, x[train, :])
		if isnothing(scaler_x) == false
            x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
            x_train = Matrix(scale_transform(x_scaler, Matrix(x[train, :]))')
        else
            x_train = Matrix(Matrix(x[train, :])')
        end
        y_train = Matrix(y[train]')
		data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
    	j = 1
        while j < (n_epochs + 1)
            my_custom_train!(flux_model, loss, loss_init, data, optimizer)
            train_loss = loss(flux_model, loss_init, x_train, y_train)
            println("epoch = " * string(j) * " training_loss = " * string(train_loss))
            j += 1
        end
        flux_model_pred = flux_model_builder
        Flux.loadmodel!(flux_model_pred, flux_model);
        y_pred_train = flux_model_pred(x_train)
		append!(y_meta, y_pred_train)
		r2_train = round((Statistics.cor(y_train[1, :], y_pred_train[1, :]))^2, digits = r_squared_precision)
        rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
		model_perform = [k r2_train rmse_train]
        if k == 1
            CSV.write(save_trained_model_at * "/stacking_model_training_records.csv", DataFrame(model_perform, [:train_iter, :r_squared_train, :rmse_train]))
        else
            CSV.write(save_trained_model_at * "/stacking_model_training_records.csv", DataFrame(model_perform, [:train_iter, :r_squared_train, :rmse_train]), append = true)
        end
		k += 1
	end
	# train the meta-learner
	flux_model = flux_model_builder
	if isnothing(scaler_x) == false
		x_scaler = fit_scaler(scaler_x, Matrix(x_meta))
		if save_trained_model
			BSON.@save(save_trained_model_at * "/saved_ensembled_meta_Xscaler(s)/Xscaler.bson", x_scaler)
		end
		x_train = Matrix(scale_transform(x_scaler, Matrix(x_meta))')
	else
		x_train = Matrix(Matrix(x_meta)')
	end
	y_train = Matrix(y_meta')
	data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
	j = 1
	while j < (n_epochs + 1)
		my_custom_train!(flux_model, loss, loss_init, data, optimizer)
		train_loss = loss(flux_model, loss_init, x_train, y_train)
		println("epoch = " * string(j) * " training_loss = " * string(train_loss))
		j += 1
	end
	flux_model_pred = flux_model_builder
	Flux.loadmodel!(flux_model_pred, flux_model);
	y_pred_train = flux_model_pred(x_train)
	r2_train = round((Statistics.cor(y_train[1, :], y_pred_train[1, :]))^2, digits = r_squared_precision)
	rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
	model_perform = [r2_train rmse_train]
	CSV.write(save_trained_model_at * "/ensembled_meta_model_training_records.csv", DataFrame(model_perform, [:r_squared_train, :rmse_train]))
	if save_trained_model
		BSON.@save(save_trained_model_at * "/saved_ensembled_meta_model(s)/trained_model_.bson", flux_model)
	end
end
