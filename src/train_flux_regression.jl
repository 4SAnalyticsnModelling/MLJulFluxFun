using Flux
using CSV
using DataFrames
using Statistics
using BSON
using MLJ
# This function evaluates flux models based on user defined resampling strategies
# cv_strategy = Cross-validation strategy (nothing means no cross-validation all data are used in training the model)
function flux_mod_eval(flux_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    cv_strategy :: Any = nothing,
    n_epochs :: Int64 = 200,
    pullback :: Bool = true,
    standardize :: Bool = false,
    lcheck :: Int64 = 5,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    loss_init = Flux.Losses.mse,
    optimizer = Flux.Optimise.Optimiser(Flux.Optimise.ClipValue(0.0001), Flux.Optimise.Optimiser(Flux.Optimise.ADAM(), Flux.Optimise.ExpDecay())))
    model_perform = Array{Float64}(undef, 0, 5)
    model_perform_mat = Array{Float64}(undef, 0, 5)
    model_perform_df = DataFrame()
    if standardize == true
        sc = MLJ.Standardizer()
    end
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    ps_init = Flux.params(flux_model)
    if isnothing(cv_strategy) == true
        train = eachindex(y)
        if standardize == true
            x_mach = MLJ.machine(sc, x[train, :])
            y_mach = MLJ.machine(sc, y[train])
            MLJ.fit!(x_mach, verbosity = 0)
            MLJ.save(save_trained_model_at * "/Xscaler.jlso", x_mach)
            MLJ.fit!(y_mach, verbosity = 0)
            MLJ.save(save_trained_model_at * "/Yscaler.jlso", y_mach)
            x_train = MLJ.transform(x_mach, x[train, :])
            y_train = MLJ.transform(y_mach, y[train])
            x_train = Matrix(x_train)'
        else
            x_train = Matrix(x[train, :])'
            y_train = y[train]
        end
        data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
        for j in 1:n_epochs
            my_custom_train!(flux_model, loss, loss_init, data, optimizer)
            train_loss = loss(flux_model, loss_init, x_train, y_train)
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
                    for l in 1:(lcheck - 1)
                        my_custom_train!(flux_model2, loss, loss_init, data, optimizer)
                        train_loss_2 = loss(flux_model2, loss_init, x_train, y_train)
                        push!(train_loss_record, train_loss_2)
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
        end
        y_pred_train = vec(flux_model(x_train))
        if standardize == true
            y_train = MLJ.inverse_transform(y_mach, y_train)
            y_pred_train = MLJ.inverse_transform(y_mach, y_pred_train)
        end
        r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision)
        rmse_train = round(sqrt(Flux.Losses.mse(y_pred_train, y_train)), digits = rmse_precision)
        weights = Flux.params(Flux.cpu(flux_model))
        BSON.@save(save_trained_model_at * "/trained_model.bson", weights)
        model_perform = [r2_train rmse_train]
        model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train])
        CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df)
    else
        for k in 1:size(cv_strategy)[1]
            flux_model1 = flux_model
            Flux.loadparams!(flux_model1, ps_init)
            train, test = cv_strategy[k, ]
            if standardize == true
                x_mach = MLJ.machine(sc, x[train, :])
                y_mach = MLJ.machine(sc, y[train])
                MLJ.fit!(x_mach, verbosity = 0)
                MLJ.save(save_trained_model_at * "/Xscaler.jlso", x_mach)
                MLJ.fit!(y_mach, verbosity = 0)
                MLJ.save(save_trained_model_at * "/Yscaler.jlso", y_mach)
                x_train = MLJ.transform(x_mach, x[train, :])
                y_train = MLJ.transform(y_mach, y[train])
                x_train = Matrix(x_train)'
                x_test = MLJ.transform(x_mach, x[test, :])
                y_test = MLJ.transform(y_mach, y[test])
                x_test = Matrix(x_test)'
            else
                x_train = Matrix(x[train, :])'
                y_train = y[train]
                x_test = Matrix(x[test, :])'
                y_test = y[test]
            end
            data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
            for j in 1:n_epochs
                my_custom_train!(flux_model1, loss, loss_init, data, optimizer)
                valid_loss = loss(flux_model1, loss_init, x_test, y_test)
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
                        for l in 1:(lcheck - 1)
                            my_custom_train!(flux_model3, loss, loss_init, data, optimizer)
                            valid_loss_2 = loss(flux_model3, loss_init, x_test, y_test)
                            push!(valid_loss_record, valid_loss_2)
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
            end
            y_pred = vec(flux_model1(x_test))
            y_pred_train = vec(flux_model1(x_train))
            if standardize == true
                y_test = MLJ.inverse_transform(y_mach, y_test)
                y_pred = MLJ.inverse_transform(y_mach, y_pred)
                y_train = MLJ.inverse_transform(y_mach, y_train)
                y_pred_train = MLJ.inverse_transform(y_mach, y_pred_train)
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
        end
    end
    return model_perform_df :: DataFrame
end
