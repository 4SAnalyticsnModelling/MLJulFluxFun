# This script contains functions that are predefined to be used in other scripts.
# Created by and © 4S Analytics & Modelling Ltd.
using Flux;
using Flux.Zygote;
using MLJ;
using DataFrames;
using DataFramesMeta;
using Statistics;
# This function returns train, test data, and save the best trained model for Flux neural network models when train test data are given
function flux_model_perform_ttgiven_save_trained_mod(flux_model,
    x :: DataFrame,
    y :: Vector,
    train_ids_in,
    test_ids_in,
    save_trained_model_as :: String,
    niter :: Int64 = 500,
    n_epochs :: Int64 = 200,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    loss_measure = Flux.Losses.mse,
    optimizer = Flux.Optimise.ADAM())
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    model_predict_test_train_df = DataFrame();
    model_predict_test_train = DataFrame();
    train, test = train_ids_in, test_ids_in;
    function loss(flux_model, x, y)
        return loss_measure(vec(flux_model(x)), y)
    end
    function my_custom_train!(flux_model, loss, ps, data, optimizer)
        ps = Zygote.Params(ps)
        for d in data
            train_loss, back = Zygote.pullback(() -> loss(flux_model, d...), ps)
            gs = back(one(train_loss))
            Flux.Optimise.update!(optimizer, ps, gs)
        end
    end
    rm(save_trained_model_as, recursive = true);
    mkdir(save_trained_model_as);
    for i in 1:niter
        train, test = MLJ.partition(eachindex(y), train_size, shuffle = true);
        x_train = Matrix(x[train, :])';
        y_train = vec(y[train, :]);
        x_test = Matrix(x[test, :])';
        y_test = vec(y[test, :]);
        data = Flux.Data.DataLoader((x_train, y_train), shuffle = true, batchsize = nobs_per_batch);
        ps = Flux.params(flux_model);
        train_loss_record = [];
        for j in 1:n_epochs
            my_custom_train!(flux_model, loss, ps, data, optimizer);
            push!(train_loss_record, loss(flux_model, x_train, y_train));
            if j > 1
                if train_loss_record[j] <= train_loss_record[j-1]
                    flux_model1 = flux_model;
                    ps1 = Flux.params(flux_model1);
                    my_custom_train!(flux_model1, loss, ps1, data, optimizer);
                    if loss(flux_model1, x_train, y_train) > train_loss_record[j]
                        try
                            Flux.stop()
                        catch
                            # println("Flux train stopped since the callback threshold has been met")
                            println(string(i) * " " * string(j) * " done");
                        finally
                        end
                        break
                    else
                        continue
                    end
                end
            end
        end
        y_test = vec(y_test);
        y_pred = vec(flux_model(x_test));
        r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = 3);
        rmse_test = round(MLJ.rms(y_test, y_pred), digits = 2);
        y_train = vec(y_train);
        y_pred_train = vec(flux_model(x_train));
        r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = 3);
        rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = 2);
        weights = Flux.params(flux_model);
        BSON.@save(save_trained_model_as * "/trained_model_" * string(i) * ".bson", weights);
        model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
        model_perform[!, :r2_flag] .= ifelse.(model_perform[!, :r_squared_test] .< model_perform[!, :r_squared_train], 1, 0);
        append!(model_perform_df, model_perform);
        sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
        sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
        sort!(model_perform_df, :r2_flag, rev = true);
        if i > 1
            rm(save_trained_model_as * "/trained_model_" * string(values(model_perform_df[2, :iter])) * ".bson");
        else
            continue
        end
        model_perform_df = DataFrame(model_perform_df[1, :]);
        model_predict_test_train = vcat(DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(iter = i, idx = train, test_train_flag = "train", observed_data = y_train, predicted_data = y_pred_train));
        append!(model_predict_test_train_df, model_predict_test_train)
        model_predict_test_train_df = model_predict_test_train_df[model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter]), :];
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end;
