using Flux;
using CSV;
using DataFrames;
using Statistics;
using BSON;
# This function evaluates flux models based on user defined resampling strategies;
# cv_strategy = Cross-validation strategy (nothing means no cross-validation; all data are used in training the model)
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
    model_perform = Array{Float64}(undef, 0, 5);
    model_perform_mat = Array{Float64}(undef, 0, 5);
    model_perform_df = DataFrame();
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    if isnothing(cv_strategy)
        train = eachindex(y);
        x_train = Matrix(x[train, :])';
        y_train = vec(y[train, :]);
        data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch);
        for j in 1:n_epochs
            my_custom_train!(flux_model, loss, data, optimizer);
            println("epoch = " * string(j) * "; training_loss = " * string(loss(flux_model, x_train, y_train)))
        end
        y_train = vec(y_train);
        y_pred_train = vec(flux_model(x_train));
        r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
        rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
        weights = Flux.params(Flux.cpu(flux_model));
        BSON.@save(save_trained_model_at * "/trained_model.bson", weights);
        model_perform = [r2_train rmse_train];
        model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train]);
        CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df);
    else
        for k in 1:size(cv_strategy)[1]
            train, test = cv_strategy[k, ];
            x_train = Matrix(x[train, :])';
            y_train = vec(y[train, :]);
            x_test = Matrix(x[test, :])';
            y_test = vec(y[test, :]);
            data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch);
            for j in 1:n_epochs
                my_custom_train!(flux_model, loss, data, optimizer);
                valid_loss = loss(flux_model, x_test, y_test);
                println("epoch = " * string(j) * "; validation_loss = " * string(valid_loss))
                flux_model1 = flux_model;
                my_custom_train!(flux_model1, loss, data, optimizer);
                if valid_loss < loss(flux_model1, x_test, y_test)
                    valid_loss_record = [];
                    flux_model2 = flux_model1
                    for l in 1:(lcheck - 1)
                        my_custom_train!(flux_model2, loss, data, optimizer);
                        push!(valid_loss_record, loss(flux_model2, x_test, y_test))
                    end
                    if sum(valid_loss .< valid_loss_record) == (lcheck - 1)
                        try
                            Flux.stop()
                        catch
                        finally
                        end
                    break
#                     else
#                         continue
                    end
                end
            end
            y_test = vec(y_test);
            y_pred = vec(flux_model(x_test));
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(rmse(y_test, y_pred), digits = rmse_precision);
            y_train = vec(y_train);
            y_pred_train = vec(flux_model(x_train));
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
            model_perform = [k r2_test r2_train rmse_test rmse_train];
            if (k == 1)
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]));
            else
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true);
            end
            model_perform_mat = vcat(model_perform_mat, model_perform)
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[4], x[5]), rev = false);
            model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[2], x[3]), rev = true);
            model_perform_mat = model_perform_mat[1, :]'
            model_perform_df = DataFrame(model_perform_mat, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]);
        end
    end
    return model_perform_df :: DataFrame
end
