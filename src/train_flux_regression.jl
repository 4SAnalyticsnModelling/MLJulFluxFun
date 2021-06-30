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
    lcheck :: Int64 = 10,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    optimizer = Flux.Optimise.ADAM())
    model_perform = Array{Float64}(undef, 0, 5)
    model_perform_mat = Array{Float64}(undef, 0, 5)
    model_perform_df = DataFrame()
    rm(save_trained_model_at, force = true, recursive = true)
    mkdir(save_trained_model_at)
    if isnothing(cv_strategy)
        train = eachindex(y)
        x_train = Matrix(x[train, :])'
        y_train = vec(y[train, :])
        data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
        for j in 1:n_epochs
            my_custom_train!(flux_model, loss, data, optimizer)
            println("epoch = " * string(j) * " training_loss = " * string(loss(flux_model, x_train, y_train)))
        end
        y_train = vec(y_train)
        y_pred_train = vec(flux_model(x_train))
        r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision)
        rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision)
        weights = Flux.params(Flux.cpu(flux_model))
        BSON.@save(save_trained_model_at * "/trained_model.bson", weights)
        model_perform = [r2_train rmse_train]
        model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train])
        CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df)
    else
        epoch_collect_max = []
        for k in 1:size(cv_strategy)[1]
            flux_model1 = flux_model
            train, test = cv_strategy[k, ]
            x_train = Matrix(x[train, :])'
            y_train = vec(y[train, :])
            x_test = Matrix(x[test, :])'
            y_test = vec(y[test, :])
            data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch)
            epoch_collect = []
            for j in 1:n_epochs
                my_custom_train!(flux_model1, loss, data, optimizer)
                valid_loss = loss(flux_model1, x_test, y_test)
                valid_r2 = round(Statistics.cor(y_test, vec(flux_model1(x_test)))^2.0, digits = r_squared_precision)
                println("epoch = " * string(j) * " validation_loss = " * string(valid_loss) * " validation_r2 = " * string(valid_r2))
                flux_model2 = flux_model1
                my_custom_train!(flux_model2, loss, data, optimizer)
                valid_loss_1 = loss(flux_model2, x_test, y_test)
                valid_r2_1 = round(Statistics.cor(y_test, vec(flux_model2(x_test)))^2.0, digits = r_squared_precision)
                if (valid_loss < valid_loss_1) & (valid_r2 > valid_r2_1)
                    valid_loss_record = []
                    valid_r2_record = []
                    flux_model3 = flux_model2
                    for l in 1:(lcheck - 1)
                        my_custom_train!(flux_model3, loss, data, optimizer)
                        valid_loss_2 = loss(flux_model3, x_test, y_test)
                        valid_r2_2 = round(Statistics.cor(y_test, vec(flux_model3(x_test)))^2.0, digits = r_squared_precision)
                        push!(valid_loss_record, valid_loss_2)
                        push!(valid_r2_record, valid_r2_2)
                    end
                    if (sum(valid_loss .< valid_loss_record) == (lcheck - 1)) & (sum(valid_r2 .> valid_r2_record) == (lcheck - 1))
                        try
                            Flux.stop()
                        catch
                        finally
                        end
                    break
                    end
                end
            push!(epoch_collect, j)
            end
            y_test = vec(y_test)
            y_pred = vec(flux_model1(x_test))
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision)
            rmse_test = round(rmse(y_test, y_pred), digits = rmse_precision)
            y_train = vec(y_train)
            y_pred_train = vec(flux_model1(x_train))
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision)
            rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision)
            model_perform = [k r2_test r2_train rmse_test rmse_train]
            if (k == 1)
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]))
            else
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true)
            end
            model_perform_mat = vcat(model_perform_mat, model_perform)
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[4], x[5]), rev = false)
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[2], x[3]), rev = true)
            model_perform_mat = model_perform_mat[1, :]'
            model_perform_df = DataFrame(model_perform_mat, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train])
        push!(epoch_collect_max, extrema(epoch_collect)[2])            
        end
    end
    return model_perform_df :: DataFrame, epoch_collect_max
end
