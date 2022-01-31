
using Flux
using CSV
using DataFrames
using Statistics
using BSON
using Flux.Zygote
# This function evaluates flux models based on user defined resampling strategies
# cv_strategy = Cross-validation strategy (nothing means no cross-validation all data are used in training the model)

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
    cv_strategy :: Any = nothing,
    n_epochs :: Int64 = 200,
    pullback :: Bool = true,
	save_trained_model :: Bool = false,
    scaler_x :: Any = nothing,
    lcheck :: Int64 = 5,
    l2_value :: Float64 = 0.0,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    loss_init = Flux.Losses.mse,
    optimizer = Flux.Optimise.Optimiser(Flux.Optimise.ADAM(), Flux.Optimise.ExpDecay()))
    # model_perform = Array{Float64}(undef, 0, 5)
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
	if save_trained_model
		mkdir(save_trained_model_at * "/saved_trained_Xscaler(s)")
		mkdir(save_trained_model_at * "/saved_trained_model(s)")
	end
    if isnothing(cv_strategy) == true
        model_perform = Array{Float64}(undef, 0, 2)
        train = eachindex(y)
        if isnothing(scaler_x) == false
            x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
			if save_trained_model
            	BSON.@save(save_trained_model_at * "/saved_trained_Xscaler(s)/Xscaler.bson", x_scaler)
			end
            x_train = Matrix(scale_transform(x_scaler, Matrix(x[train, :]))')
        else
            x_train = Matrix(Matrix(x[train, :])')
        end
        y_train = Matrix(y[train]')
        data = Flux.Data.DataLoader((x_train, y_train), batchsize = nobs_per_batch, shuffle = true)
        flux_model = flux_model_builder
        j = 1
        while j < (n_epochs + 1)
            my_custom_train!(flux_model, loss, loss_init, data, optimizer, lambda2(l2_value))
            train_loss = loss(flux_model, loss_init, x_train, y_train, lambda2(l2_value))
            if isnan(train_loss) == false
                println("epoch = " * string(j) * " training_loss = " * string(train_loss))
            else
                try
                    Flux.skip()
                catch
                finally
                end
            end
            j += 1
        end
        y_pred_train = flux_model(x_train)
        r2_train = round((Statistics.cor(y_train[1, :], y_pred_train[1, :]))^2, digits = r_squared_precision)
        rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
		if save_trained_model
        	weights = Flux.params(Flux.cpu(flux_model))
        	BSON.@save(save_trained_model_at * "/saved_trained_model(s)/trained_model.bson", weights)
		end
        model_perform = [r2_train rmse_train]
        model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train])
        CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df)
    else
        model_perform = Array{Float64}(undef, 0, 5)
        k = 1
        while k < (1 + size(cv_strategy)[1])
            if pullback == true
                valid_loss_record = []
                early_stop_flag = 0
                params_dict = Dict()
            end
            flux_model = flux_model_builder
            train, test = cv_strategy[k, ]
            if isnothing(scaler_x) == false
                x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
				if save_trained_model
                	BSON.@save(save_trained_model_at * "/saved_trained_Xscaler(s)/Xscaler_" * string(k) * ".bson", x_scaler)
				end
                x_train = Matrix(scale_transform(x_scaler, Matrix(x[train, :]))')
                x_test = Matrix(scale_transform(x_scaler, Matrix(x[test, :]))')
            else
                x_train = Matrix(Matrix(x[train, :])')
                x_test = Matrix(Matrix(x[test, :])')
            end
            y_train = Matrix(y[train]')
            y_test = Matrix(y[test]')
            data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
            j = 1
            while j < (n_epochs + 1)
                my_custom_train!(flux_model, loss, loss_init, data, optimizer, lambda2(l2_value))
                valid_loss = loss(flux_model, loss_init, x_test, y_test, lambda2(l2_value))
                push!(valid_loss_record, valid_loss)
                push!(params_dict, Symbol("weights" * string(j)) => Flux.params(Flux.cpu(flux_model)))
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
                weights = params_dict[Symbol("weights" * string(j - early_stop_flag))]
            else
                weights = params_dict[Symbol("weights" * string(n_epochs))]
            end
            flux_model_pred = flux_model_builder
            Flux.loadparams!(flux_model_pred, weights);
            y_pred = flux_model_pred(x_test)
            y_pred_train = flux_model_pred(x_train)
            r2_test = round((Statistics.cor(y_test[1, :], y_pred[1, :]))^2, digits = r_squared_precision)
            rmse_test = round(sqrt(Flux.Losses.mse(y_pred, y_test)), digits = rmse_precision)
            r2_train = round((Statistics.cor(y_train[1, :], y_pred_train[1, :]))^2, digits = r_squared_precision)
            rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
            model_perform = [k r2_test r2_train rmse_test rmse_train]
            if k == 1
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]))
            else
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true)
            end
			if save_trained_model
				weights = Flux.params(Flux.cpu(flux_model))
				BSON.@save(save_trained_model_at * "/saved_trained_model(s)/trained_model_" * string(k) * ".bson", weights)
			end
            k += 1
        end
    end
end
