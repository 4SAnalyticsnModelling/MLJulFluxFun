using Flux
using CSV
using DataFrames
using Statistics
using BSON
# This function evaluates flux models based on user defined resampling strategies
# cv_strategy = Cross-validation strategy (nothing means no cross-validation all data are used in training the model)
function flux_mod_eval(flux_model,
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
    optimizer = Flux.Optimise.Optimiser(Flux.Optimise.ClipValue(0.0001), Flux.Optimise.Optimiser(Flux.Optimise.ADAM(), Flux.Optimise.ExpDecay())))
    model_perform = Array{Float64}(undef, 0, 5)
    model_perform_mat = Array{Float64}(undef, 0, 5)
    model_perform_df = DataFrame()
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    ps_init = Flux.params(flux_model)
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
        j = 1
        while j < (n_epochs + 1)
            my_custom_train!(flux_model, loss, loss_init, data, optimizer)
            train_loss = loss(flux_model, loss_init, x_train, y_train)
            if isnan(train_loss) == false
                ps = Flux.params(flux_model)
                println("epoch = " * string(j) * " training_loss = " * string(train_loss))
                if pullback == true
                    flux_model1 = flux_model
                    Flux.loadparams!(flux_model1, ps)
                    my_custom_train!(flux_model1, loss, loss_init, data, optimizer)
                    train_loss_1 = loss(flux_model1, loss_init, x_train, y_train)
                    ps1 = Flux.params(flux_model1)
                    if train_loss < train_loss_1
                        train_loss_record = []
                        flux_model2 = flux_model
                        Flux.loadparams!(flux_model2, ps1)
                        l = 1
                        while l < lcheck
                            my_custom_train!(flux_model2, loss, loss_init, data, optimizer)
                            train_loss_2 = loss(flux_model2, loss_init, x_train, y_train)
                            push!(train_loss_record, train_loss_2)
                            l += 1
                        end
                        if sum(train_loss .< train_loss_record) == (lcheck - 1)
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
            y_train = vec(scale_back(y_scaler, Matrix(reshape(y_train, length(train), 1)))[1, :])
            y_pred_train = vec(scale_back(y_scaler, Matrix(reshape(y_pred_train, length(train), 1)))[1, :])
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
            flux_model1 = flux_model
            Flux.loadparams!(flux_model1, ps_init)
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
                my_custom_train!(flux_model1, loss, loss_init, data, optimizer)
                valid_loss = loss(flux_model1, loss_init, x_test, y_test)
                if isnan(valid_loss) == false
                    println("epoch = " * string(j) * " validation_loss = " * string(valid_loss))
                    ps1 = Flux.params(flux_model1)
                    if pullback == true
                        flux_model2 = flux_model1
                        Flux.loadparams!(flux_model2, ps1)
                        my_custom_train!(flux_model2, loss, loss_init, data, optimizer)
                        valid_loss_1 = loss(flux_model2, loss_init, x_test, y_test)
                        ps2 = Flux.params(flux_model2)
                        if valid_loss < valid_loss_1
                            valid_loss_record = []
                            flux_model3 = flux_model2
                            Flux.loadparams!(flux_model3, ps2)
                            l = 1
                            while l < lcheck
                                my_custom_train!(flux_model3, loss, loss_init, data, optimizer)
                                valid_loss_2 = loss(flux_model3, loss_init, x_test, y_test)
                                push!(valid_loss_record, valid_loss_2)
                                l += 1
                            end
                            if sum(valid_loss .< valid_loss_record) == (lcheck - 1)
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
            y_pred = vec(flux_model1(x_test)[1, :])
            y_pred_train = vec(flux_model1(x_train)[1, :])
            if isnothing(scaler_y) == false
                y_train = vec(scale_back(y_scaler, Matrix(reshape(y_train, length(train), 1)))[1, :])
                y_pred_train = vec(scale_back(y_scaler, Matrix(reshape(y_pred_train, length(train), 1)))[1, :])
                y_test = vec(scale_back(y_scaler, Matrix(reshape(y_test, length(test), 1)))[1, :])
                y_pred = vec(scale_back(y_scaler, Matrix(reshape(y_pred, length(test), 1)))[1, :])
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
