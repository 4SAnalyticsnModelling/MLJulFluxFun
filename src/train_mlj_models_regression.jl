using MLJ;
using XGBoost;
using DecisionTree;
using DataFrames;
using Statistics;
# This function returns train, test ids, and save the best trained model for MLJ models when train test data are not pre-split
function mlj_mod_train(mlj_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_size :: Float64 = 0.75,
    niter :: Int64 = 500,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    model_predict_test_train_df = DataFrame();
    model_predict_test_train = DataFrame();
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            train, test = MLJ.partition(eachindex(y), train_size, shuffle = true);
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            MLJ.save(save_trained_model_at * "/trained_model_" * string(i) * "_" * string(tuning_param) * ".jlso", mach, compression = :none);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_perform[!, :r2_flag] .= ifelse.(isnan.(model_perform[!, :r_squared_test]) .& isnan.(model_perform[!, :r_squared_train]), -3,
            ifelse.(isnan.(model_perform[!, :r_squared_test]), -2,
            ifelse.(isnan.(model_perform[!, :r_squared_train]), -1,
            ifelse.(model_perform[!, :r_squared_test] .< model_perform[!, :r_squared_train], 1, 0))));
            model_perform[!, tuning_param_label] .= tuning_param;
            append!(model_perform_df, model_perform);
            sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
            sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
            sort!(model_perform_df, :r2_flag, rev = true);
            if size(model_perform_df)[1] > 1
                rm(save_trained_model_at * "/trained_model_" * string(values(model_perform_df[2, :iter])) * "_" * string(values(model_perform_df[2, tuning_param_label])) * ".jlso", force = true);
                rm(save_trained_model_at * "/trained_model_" * string(values(model_perform_df[2, :iter])) * "_" * string(values(model_perform_df[2, tuning_param_label])) * ".jlso", force = true);
            end
            model_perform_df = DataFrame(model_perform_df[1, :]);
            model_predict_test_train = vcat(DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(iter = i, idx = train, test_train_flag = "train", observed_data = y_train, predicted_data = y_pred_train));
            model_predict_test_train[!, tuning_param_label] .= tuning_param;
            append!(model_predict_test_train_df, model_predict_test_train)
            model_predict_test_train_df = model_predict_test_train_df[(model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter])) .& (model_predict_test_train_df[!, tuning_param_label] .== values(model_perform_df[1, tuning_param_label])), :];
        end
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end
# This function returns train, test ids, and save the best trained model for MLJ models when train test data are pre-split
function mlj_mod_train_tt(mlj_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    train_ids_in,
    test_ids_in,
    train_sample_fraction :: Float64 = 1.0,
    test_sample_fraction :: Float64 = 1.0,
    niter :: Int64 = 500,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    model_perform_df = DataFrame();
    model_perform = DataFrame();
    model_predict_test_train_df = DataFrame();
    model_predict_test_train = DataFrame();
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            if train_sample_fraction < 1.0
                train = MLJ.partition(train_ids_in, train_sample_fraction, shuffle = true)[1]
            else
                train = train_ids_in
            end
            if test_sample_fraction < 1.0
                test = MLJ.partition(test_ids_in, test_sample_fraction, shuffle = true)[1]
            else
                test = test_ids_in
            end
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            MLJ.save(save_trained_model_at * "/trained_model_" * string(i) * "_" * string(tuning_param) * ".jlso", mach, compression = :none);
            y_pred = MLJ.predict(mach, rows = test);
            y_pred_train = MLJ.predict(mach, rows = train);
            y_train = vec(y[train, :]);
            y_test = vec(y[test, :]);
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
            rmse_test = round(MLJ.rms(y_test, y_pred), digits = rmse_precision);
            rmse_train = round(MLJ.rms(y_train, y_pred_train), digits = rmse_precision);
            model_perform = DataFrame(iter = i, r_squared_test = r2_test, r_squared_train = r2_train, rmse_test = rmse_test, rmse_train = rmse_train);
            model_perform[!, :r2_flag] .= ifelse.(isnan.(model_perform[!, :r_squared_test]) .& isnan.(model_perform[!, :r_squared_train]), -3,
            ifelse.(isnan.(model_perform[!, :r_squared_test]), -2,
            ifelse.(isnan.(model_perform[!, :r_squared_train]), -1,
            ifelse.(model_perform[!, :r_squared_test] .< model_perform[!, :r_squared_train], 1, 0))));
            model_perform[!, tuning_param_label] .= tuning_param;
            append!(model_perform_df, model_perform);
            sort!(model_perform_df, [:rmse_test, :rmse_train], rev = false);
            sort!(model_perform_df, [:r_squared_test, :r_squared_train], rev = true);
            sort!(model_perform_df, :r2_flag, rev = true);
            if size(model_perform_df)[1] > 1
                rm(save_trained_model_at * "/trained_model_" * string(values(model_perform_df[2, :iter])) * "_" * string(values(model_perform_df[2, tuning_param_label])) * ".jlso", force = true);
                rm(save_trained_model_at * "/trained_model_" * string(values(model_perform_df[2, :iter])) * "_" * string(values(model_perform_df[2, tuning_param_label])) * ".jlso", force = true);
            end
            model_perform_df = DataFrame(model_perform_df[1, :]);
            model_predict_test_train = vcat(DataFrame(iter = i, idx = test, test_train_flag = "test", observed_data = y_test, predicted_data = y_pred), DataFrame(iter = i, idx = train, test_train_flag = "train", observed_data = y_train, predicted_data = y_pred_train));
            model_predict_test_train[!, tuning_param_label] .= tuning_param;
            append!(model_predict_test_train_df, model_predict_test_train)
            model_predict_test_train_df = model_predict_test_train_df[(model_predict_test_train_df[!, :iter] .== values(model_perform_df[1, :iter])) .& (model_predict_test_train_df[!, tuning_param_label] .== values(model_perform_df[1, tuning_param_label])), :];
        end
    end
    return model_perform_df :: DataFrame, model_predict_test_train_df :: DataFrame
end
