# This script contains functions that are predefined to be used in other scripts.
# Created by and © 4S Analytics & Modelling Ltd.

# This function returns train, test data, model performance and SHAP values for MLJ models
function mlj_model_shap(mlj_model,
    x :: DataFrame,
    y :: Vector,
    explain :: DataFrame,
    reference :: DataFrame,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_size :: Float64 = 0.75,
    niter :: Int64 = 500,
    nrepeats :: Int64 = 10,
    shap_sample_size :: Int64 = 60,
    shap_seed :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    shap_precision :: Int64 = 4)
# Start calculation
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    train_idx = DataFrame();
    test_idx = DataFrame();
    train_idx_df = DataFrame();
    test_idx_df = DataFrame();
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            train, test = partition(eachindex(y), train_size, shuffle = true);
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_perform[!, tuning_param_label] .= tuning_param;
            append!(model_perform_df, model_perform);
            model_perform_df[!, :r2_flag] .= ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0);
            model_perform_df = DataFrame(sort!(sort!(sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false), [:r_squared_test, :r_squared_train], rev = true), :r2_flag, rev = true)[1, :])[:, Not(:r2_flag)];
            train_idx = DataFrame(iter = i, train_idx = train);
            train_idx[!, tuning_param_label] .= tuning_param;
            append!(train_idx_df, train_idx)
            test_idx = DataFrame(iter = i, test_idx = test);
            test_idx[!, tuning_param_label] .= tuning_param;
            append!(test_idx_df, test_idx)
            train_idx_df = @where(train_idx_df, :iter .== values(model_perform_df.iter[1]));
            test_idx_df = @where(test_idx_df, :iter .== values(model_perform_df.iter[1]));
        end
    end
    show(model_perform_df, allcols = true)
    model_perform_df_f = DataFrame();
    model_predict_test_train_df = DataFrame();
    global_shap_values_df = DataFrame();
    local_shap_values_df = DataFrame();
    tuning_param_name = values(test_idx_df[1, tuning_param_label]);
    train, test = train_idx_df.train_idx, test_idx_df.test_idx;
    function predict_function(model, data)
      return DataFrame(y_pred = MLJ.predict(model, data))
    end
    for i in 1:niter*nrepeats
        if i > 0.5*niter*nrepeats
            train, test = MLJ.partition(eachindex(y), train_size, shuffle = true);
        else
            train, test = train_idx_df.train_idx, test_idx_df.test_idx;
        end
        mach = MLJ.machine(mlj_model, x, y);
        MLJ.fit!(mach, rows = train, verbosity = 0);
        y_pred = MLJ.predict(mach, rows = test);
        y_pred_train = MLJ.predict(mach, rows = train);
        y_train = vec(y[train, :]);
        y_test = vec(y[test, :]);
        r2_train = round((cor(y_train, y_pred_train))^2, digits = r_squared_precision);
        r2_test = round((cor(y_test, y_pred))^2, digits = r_squared_precision);
        if (r2_train > r2_test) & (r2_train >= values(model_perform_df.r_squared_train[1])) & (r2_test >= values(model_perform_df.r_squared_test[1]))
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform_df_f = DataFrame(r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_predict_test_train_df = vcat(DataFrame(idx = test, flag1 = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(idx = train, flag1 = "train", observed_data = y_train, predicted_data = y_pred_train));
            local_shap_values_df = ShapML.shap(explain = explain,
                            reference = reference,
                            model = mach,
                            predict_function = predict_function,
                            sample_size = shap_sample_size,
                            seed = shap_seed);
            global_shap_values_df = combine(groupby(local_shap_values_df, [:feature_name]), mean_absolute_shap_values = :shap_effect =>
            xq -> abs(round(mean(xq), digits = shap_precision)));
            local_shap_values_df[!, :absolute_shap_values] .= round.(abs.(local_shap_values_df[!, :shap_effect]), digits = shap_precision)
            local_shap_values_df = local_shap_values_df[!, [:index, :feature_name, :absolute_shap_values]]
            break
        end
    end
    return model_perform_df_f :: DataFrame, model_predict_test_train_df :: DataFrame, local_shap_values_df :: DataFrame, global_shap_values_df :: DataFrame
end

# This function returns model performance and SHAP values for MLJ models when train test indices are given
function mlj_model_shap_ttgiven(mlj_model,
    x :: DataFrame,
    y :: Vector,
    explain :: DataFrame,
    reference :: DataFrame,
    tuning_param_name,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_ids_in,
    test_ids_in,
    niter :: Int64 = 500,
    nrepeats :: Int64 = 10,
    shap_sample_size :: Int64 = 60,
    shap_seed :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    shap_precision :: Int64 = 4)
# Start calculation
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    train_idx = DataFrame();
    test_idx = DataFrame();
    train_idx_df = DataFrame();
    test_idx_df = DataFrame();
    train, test = train_ids_in, test_ids_in;
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_perform[!, tuning_param_label] .= tuning_param;
            append!(model_perform_df, model_perform);
            model_perform_df[!, :r2_flag] .= ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0);
            model_perform_df = DataFrame(sort!(sort!(sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false), [:r_squared_test, :r_squared_train], rev = true), :r2_flag, rev = true)[1, :])[:, Not(:r2_flag)];
            train_idx = DataFrame(iter = i, train_idx = train);
            train_idx[!, tuning_param_label] .= tuning_param;
            append!(train_idx_df, train_idx)
            test_idx = DataFrame(iter = i, test_idx = test);
            test_idx[!, tuning_param_label] .= tuning_param;
            append!(test_idx_df, test_idx)
            train_idx_df = @where(train_idx_df, :iter .== values(model_perform_df.iter[1]));
            test_idx_df = @where(test_idx_df, :iter .== values(model_perform_df.iter[1]));
        end
    end
    model_perform_df_f = DataFrame();
    model_predict_test_train_df = DataFrame();
    global_shap_values_df = DataFrame();
    local_shap_values_df = DataFrame();
    tuning_param_name = values(test_idx_df[1, tuning_param_label]);
    function predict_function(model, data)
      return DataFrame(y_pred = MLJ.predict(model, data))
    end
    for i in 1:niter*nrepeats
        mach = MLJ.machine(mlj_model, x, y);
        MLJ.fit!(mach, rows = train, verbosity = 0);
        y_pred = MLJ.predict(mach, rows = test);
        y_pred_train = MLJ.predict(mach, rows = train);
        y_train = vec(y[train, :]);
        y_test = vec(y[test, :]);
        r2_train = round((cor(y_train, y_pred_train))^2, digits = r_squared_precision);
        r2_test = round((cor(y_test, y_pred))^2, digits = r_squared_precision);
        if (r2_train > r2_test) & (r2_train >= values(model_perform_df.r_squared_train[1])) & (r2_test >= values(model_perform_df.r_squared_test[1]))
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform_df_f = DataFrame(r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_predict_test_train_df = vcat(DataFrame(idx = test, flag1 = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(idx = train, flag1 = "train", observed_data = y_train, predicted_data = y_pred_train));
            local_shap_values_df = ShapML.shap(explain = explain,
                            reference = reference,
                            model = mach,
                            predict_function = predict_function,
                            sample_size = shap_sample_size,
                            seed = shap_seed);
            global_shap_values_df = combine(groupby(local_shap_values_df, [:feature_name]), mean_absolute_shap_values = :shap_effect =>
            xq -> abs(round(mean(xq), digits = shap_precision)));
            local_shap_values_df[!, :absolute_shap_values] .= round.(abs.(local_shap_values_df[!, :shap_effect]), digits = shap_precision)
            local_shap_values_df = local_shap_values_df[!, [:index, :feature_name, :absolute_shap_values]]
            break
        end
    end
    return model_perform_df_f :: DataFrame, model_predict_test_train_df :: DataFrame, local_shap_values_df :: DataFrame, global_shap_values_df :: DataFrame
end

# This function returns model performance for MLJ models when train test indices are given
function mlj_model_perform_ttgiven(mlj_model,
    x :: DataFrame,
    y :: Vector,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_ids_in,
    test_ids_in,
    niter :: Int64 = 500,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
# Start calculation
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    model_predict_test_train_df = DataFrame();
    model_predict_test_train = DataFrame();
    train, test = train_ids_in, test_ids_in;
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_perform[!, tuning_param_label] .= tuning_param;
            append!(model_perform_df, model_perform);
            model_perform_df[!, :r2_flag] .= ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0);
            sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
            sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
            sort!(model_perform_df, :r2_flag, rev = true);
            model_perform_df = DataFrame(model_perform_df[1, Not(:r2_flag)]);
            model_predict_test_train = vcat(DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(iter = i, idx = train, test_train_flag = "train", observed_data = y_train, predicted_data = y_pred_train));
            model_predict_test_train[!, tuning_param_label] .= tuning_param;
            append!(model_predict_test_train_df, model_predict_test_train)
            model_predict_test_train_df = model_predict_test_train_df[(model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter])) .& (model_predict_test_train_df[!, tuning_param_label] .== values(model_perform_df[1, tuning_param_label])), :];
        end
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end

# This function returns model performance for MLJ models when train test indices are given and saves the trained model
function mlj_model_perform_ttgiven_save_trained_mod(mlj_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_as :: String,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_ids_in,
    test_ids_in,
    niter :: Int64 = 500,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
# Start calculation
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    model_predict_test_train_df = DataFrame();
    model_predict_test_train = DataFrame();
    rm(save_trained_model_as, recursive = true);
    mkdir(save_trained_model_as);
    train, test = train_ids_in, test_ids_in;
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
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
end
end

# This function returns train, test data, model performance and SHAP values for Flux neural network models
function flux_model_shap(flux_model,
    x :: DataFrame,
    y :: Vector,
    explain :: DataFrame,
    reference :: DataFrame,
    train_size :: Float64 = 0.75,
    niter :: Int64 = 500,
    nrepeats :: Int64 = 10,
    n_epochs :: Int64 = 200,
    nobs_per_batch :: Int64 = 1,
    shap_sample_size :: Int64 = 60,
    shap_seed :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    shap_precision :: Int64 = 4,
    loss_measure = Flux.Losses.mse,
    optimizer = Flux.Optimise.ADAM())
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    train_idx = DataFrame();
    test_idx = DataFrame();
    train_idx_df = DataFrame();
    test_idx_df = DataFrame();
    function loss(flux_model, x, y)
        return loss_measure(flux_model(x), y)
    end
    function my_custom_train!(flux_model, loss, ps, data, optimizer)
        ps = Zygote.Params(ps)
        for d in data
            train_loss, back = Zygote.pullback(() -> loss(flux_model, d...), ps)
            gs = back(one(train_loss))
            Flux.Optimise.update!(optimizer, ps, gs)
        end
    end
    for i in 1:niter
        train, test = MLJ.partition(eachindex(y), train_size, shuffle = true);
        x_train = Matrix(Array(x[train, :])');
        y_train = Matrix(Array(y[train, :])');
        x_test = Matrix(Array(x[test, :])');
        y_test = Matrix(Array(y[test, :])');
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
                            println("Flux train stopped since the callback threshold has been met")
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
        model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
        append!(model_perform_df, model_perform);
        model_perform_df[!, :r2_flag] .= ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0);
        model_perform_df = DataFrame(DataFrame(sort!(sort!(sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false), [:r_squared_test, :r_squared_train], rev = true),:r2_flag, rev = true))[1, Not(:r2_flag)]);
        train_idx = DataFrame(iter = i, train_idx = train);
        append!(train_idx_df, train_idx)
        test_idx = DataFrame(iter = i, test_idx = test);
        append!(test_idx_df, test_idx)
        train_idx_df = @where(train_idx_df, :iter .== values(model_perform_df.iter[1]));
        test_idx_df = @where(test_idx_df, :iter .== values(model_perform_df.iter[1]));
    end
        model_perform_df_f = DataFrame();
        model_predict_test_train_df = DataFrame();
        global_shap_values_df = DataFrame();
        local_shap_values_df = DataFrame();
        for i in 1:niter*nrepeats
            if i > 0.5*niter*nrepeats
                train, test = MLJ.partition(eachindex(y), train_size, shuffle = true);
            else
                train, test = train_idx_df.train_idx, test_idx_df.test_idx;
            end
            x_train = Matrix(Array(x[train, :])');
            y_train = Matrix(Array(y[train, :])');
            x_test = Matrix(Array(x[test, :])');
            y_test = Matrix(Array(y[test, :])');
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
                                println("Flux train stopped since the callback threshold has been met")
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
            y_train = vec(y_train);
            y_pred_train = vec(flux_model(x_train));
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = 3);
            if (r2_train > r2_test) & (r2_train >= values(model_perform_df.r_squared_train[1])) & (r2_test >= values(model_perform_df.r_squared_test[1]))
                rmse_test = round(MLJ.rms(y_test, y_pred), digits = 2);
                rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = 2);
                model_perform_df_f = DataFrame(r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
                model_predict_test_train_df = vcat(DataFrame(idx = test, flag1 = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(idx = train, flag1 = "train", observed_data = y_train, predicted_data = y_pred_train));
                function predict_function(model, data)
                  return DataFrame(y_pred = vec(model(Matrix(Array(data)'))))
                end
                local_shap_values_df = ShapML.shap(explain = explain,
                                reference = reference,
                                model = flux_model,
                                predict_function = predict_function,
                                sample_size = shap_sample_size,
                                seed = shap_seed);
                global_shap_values_df = combine(groupby(local_shap_values_df, [:feature_name]), mean_absolute_shap_values = :shap_effect =>
                xq -> abs(round(mean(xq), digits = shap_precision)));
                local_shap_values_df[!, :absolute_shap_values] .= round.(abs.(local_shap_values_df[!, :shap_effect]), digits = shap_precision)
                local_shap_values_df = local_shap_values_df[!, [:index, :feature_name, :absolute_shap_values]]
                break
            else
                continue
            end
        end
    return model_perform_df_f :: DataFrame, model_predict_test_train_df :: DataFrame, local_shap_values_df :: DataFrame, global_shap_values_df :: DataFrame
end;


# This function returns train, test data, and model performance for Flux neural network models when train test data are given
function flux_model_perform_ttgiven(flux_model,
    x :: DataFrame,
    y :: Vector,
    train_ids_in,
    test_ids_in,
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
    for i in 1:niter
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
                            println("Flux train stopped since the callback threshold has been met")
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
        model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
        append!(model_perform_df, model_perform);
        model_perform_df[!, :r2_flag] .= ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0);
        sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
        sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
        sort!(model_perform_df, :r2_flag, rev = true);
        model_perform_df = DataFrame(model_perform_df[1, :])[:, Not(:r2_flag)];
        model_predict_test_train = vcat(DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(iter = i, idx = train, test_train_flag = "train", observed_data = y_train, predicted_data = y_pred_train));
        append!(model_predict_test_train_df, model_predict_test_train)
        model_predict_test_train_df = model_predict_test_train_df[model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter]), :];
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end;

# This function returns test data, and model performance for Flux neural network models when train test data are given
function flux_model_perform_ttgiven_no_train(flux_model,
    x :: DataFrame,
    y :: Vector,
    train_ids_in,
    test_ids_in,
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
    for i in 1:niter
        x_train = Matrix(x[train, :])';
        y_train = vec(y[train, :]);
        x_test = Matrix(x[test, :])';
        y_test = vec(y[test, :]);
        # x_train = Matrix(Array(x[train, :])');
        # y_train = Matrix(Array(y[train, :])');
        # x_test = Matrix(Array(x[test, :])');
        # y_test = Matrix(Array(y[test, :])');
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
                            println("Flux train stopped since the callback threshold has been met")
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
        model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
        append!(model_perform_df, model_perform);
        model_perform_df[!, :r2_flag] .= ifelse.(isnan.(model_perform_df[!, :r_squared_test]) .| isnan.(model_perform_df[!, :r_squared_train]), -1, ifelse.(model_perform_df[!, :r_squared_test] .< model_perform_df[!, :r_squared_train], 1, 0));
        sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
        sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
        sort!(model_perform_df, :r2_flag, rev = true);
        model_perform_df = DataFrame(model_perform_df[1, :])[:, Not(:r2_flag)];
        model_predict_test_train = DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred);
        append!(model_predict_test_train_df, model_predict_test_train)
        model_predict_test_train_df = model_predict_test_train_df[model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter]), :];
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end;

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

# This function returns train, test data, and save the best trained model for Flux neural network models when train test data are not given
function flux_model_perform_save_trained_mod(flux_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_as :: String,
    train_size :: Float64 = 0.75,
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
