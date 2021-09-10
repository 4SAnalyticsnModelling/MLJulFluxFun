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
    scaler_x :: Any = nothing,
    scaler_y :: Any = nothing,
    lcheck :: Int64 = 5,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    loss_init = Flux.Losses.mse,
    optimizer = Flux.Optimise.Optimiser(Flux.Optimise.Optimiser(Flux.Optimise.ADAM(), Flux.Optimise.ExpDecay())))
    model_perform = Array{Float64}(undef, 0, 5)
    model_perform_mat = Array{Float64}(undef, 0, 5)
    model_perform_df = DataFrame()
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    if isnothing(cv_strategy) == true
        train = eachindex(y)
        if isnothing(scaler_x) == false
            x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
            BSON.@save(save_trained_model_at * "/Xscaler.bson", x_scaler)
            x_train = scale_transform(x_scaler, Matrix(x[train, :]))'
        else
            x_train = Matrix(x[train, :])'
        end
        if isnothing(scaler_y) == false
            y_scaler = fit_scaler(scaler_y, Matrix(reshape(y[train], length(train), 1)))
            BSON.@save(save_trained_model_at * "/Yscaler.bson", y_scaler)
            y_train = vec(scale_transform(y_scaler, Matrix(reshape(y[train], length(train), 1)))[:, 1])
        else
            y_train = y[train]
        end
        data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
        if pullback == true
            train_loss_record = []
        end
        flux_model = flux_model_builder
        j = 1
        while j < (n_epochs + 1)
            my_custom_train!(flux_model, loss, loss_init, data, optimizer)
            train_loss = loss(flux_model, loss_init, x_train, y_train)
            if isnan(train_loss) == false
                println("epoch = " * string(j) * " training_loss = " * string(train_loss))
                if pullback == true
                    push!(train_loss_record, train_loss)
                    if j > (lcheck + 1)
                        if sum(train_loss .> train_loss_record[(end - 1 - lcheck):(end - 1)]) == lcheck
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
        y_pred_train = vec(flux_model(x_train)[1, :])
        if isnothing(scaler_y) == false
            y_train = vec(scale_back(y_scaler, Matrix(reshape(y_train, length(train), 1)))[:, 1])
            y_pred_train = vec(scale_back(y_scaler, Matrix(reshape(y_pred_train, length(train), 1)))[:, 1])
        end
        r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision)
        rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
        weights = Flux.params(Flux.cpu(flux_model))
        BSON.@save(save_trained_model_at * "/trained_model.bson", weights)
        model_perform = [r2_train rmse_train]
        model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train])
        CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df)
    else
        k = 1
        while k < (1 + size(cv_strategy)[1])
            if pullback == true
                valid_loss_record = []
            end
            flux_model = flux_model_builder
            train, test = cv_strategy[k, ]
            if isnothing(scaler_x) == false
                x_scaler = fit_scaler(scaler_x, Matrix(x[train, :]))
                BSON.@save(save_trained_model_at * "/Xscaler.bson", x_scaler)
                x_train = scale_transform(x_scaler, Matrix(x[train, :]))'
                x_test = scale_transform(x_scaler, Matrix(x[test, :]))'
            else
                x_train = Matrix(x[train, :])'
                x_test = Matrix(x[test, :])'
            end
            if isnothing(scaler_y) == false
                y_scaler = fit_scaler(scaler_y, Matrix(reshape(y[train], length(train), 1)))
                BSON.@save(save_trained_model_at * "/Yscaler.bson", y_scaler)
                y_train = vec(scale_transform(y_scaler, Matrix(reshape(y[train], length(train), 1)))[:, 1])
                y_test = vec(scale_transform(y_scaler, Matrix(reshape(y[test], length(test), 1)))[:, 1])
            else
                y_train = y[train]
                y_test = y[test]
            end
            data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
            j = 1
            while j < (n_epochs + 1)
                my_custom_train!(flux_model, loss, loss_init, data, optimizer)
                valid_loss = loss(flux_model, loss_init, x_test, y_test)
                if isnan(valid_loss) == false
                    println("epoch = " * string(j) * " validation_loss = " * string(valid_loss))
                    if pullback == true
                        push!(valid_loss_record, valid_loss)
                        if j > (lcheck + 1)
                            if sum(valid_loss .> valid_loss_record[(end -1 - lcheck):(end - 1)]) == lcheck
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
            y_pred = vec(flux_model(x_test)[1, :])
            y_pred_train = vec(flux_model(x_train)[1, :])
            if isnothing(scaler_y) == false
                y_train = vec(scale_back(y_scaler, Matrix(reshape(y_train, length(train), 1)))[:, 1])
                y_pred_train = vec(scale_back(y_scaler, Matrix(reshape(y_pred_train, length(train), 1)))[:, 1])
                y_test = vec(scale_back(y_scaler, Matrix(reshape(y_test, length(test), 1)))[:, 1])
                y_pred = vec(scale_back(y_scaler, Matrix(reshape(y_pred, length(test), 1)))[:, 1])
            end
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision)
            rmse_test = round(sqrt(Flux.Losses.mse(y_pred, y_test)), digits = rmse_precision)
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision)
            rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
            weights = Flux.params(Flux.cpu(flux_model))
            BSON.@save(save_trained_model_at * "/trained_model.bson", weights)
            model_perform = [k r2_test r2_train rmse_test rmse_train]
            if k == 1
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]))
            else
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true)
            end
            model_perform_mat = vcat(model_perform_mat, model_perform)
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[4], x[5]), rev = false)
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[2], x[3]), rev = true)
            model_perform_mat = model_perform_mat[1, :]'
            model_perform_df = DataFrame(model_perform_mat, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train])
            k += 1
        end
    end
    return model_perform_df
end
