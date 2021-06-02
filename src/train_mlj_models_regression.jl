using MLJ;
using XGBoost;
using DecisionTree;
using DataFrames;
using Statistics;
# This function evaluates flux models based on user defined resampling strategies;
# cv_strategy = Cross-validation strategy (nothing means no cross-validation; all data are used in training the model)
function mlj_mod_eval(mlj_model,
    x :: DataFrame,
    y :: Vector,
    save_trained_model_at :: String,
    tuning_param_name,
    tuning_param_label :: Symbol,
    tuning_param_rng :: T,
    cv_strategy :: Any = nothing,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2) where {T <: Union{StepRange{Int64, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}
    model_perform = Array{Float64}(undef, 0, 6);
    model_perform_mat = Array{Float64}(undef, 0, 6);
    model_perform_df = DataFrame();
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    for tuning_param in tuning_param_rng
        tuning_param_name = tuning_param;
        if isnothing(cv_strategy)
            train = eachindex(y);
            mach = MLJ.machine(mlj_model, x, y);
            MLJ.fit!(mach, rows = train, verbosity = 0);
            y_train = vec(y[train, :]);
            y_pred_train = vec(MLJ.predict(mach, rows = train));
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
            MLJ.save(save_trained_model_at * "/trained_model.jlso", mach, compression = :none);
            model_perform = [tuning_param r2_train rmse_train];
            model_perform_df = DataFrame(model_perform[1, :]', [tuning_param_label, :r_squared_train, :rmse_train]);
            if tuning_param == tuning_param_rng[1]
                CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df);
            else
                CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df, append = true);
            end
        else
            for k in 1:size(cv_strategy)[1]
                train, test = cv_strategy[k, ];
                mach = MLJ.machine(mlj_model, x, y);
                MLJ.fit!(mach, rows = train, verbosity = 0);
                y_test = vec(y[test, :]);
                y_pred = vec(MLJ.predict(mach, rows = test));
                r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
                rmse_test = round(rmse(y_test, y_pred), digits = rmse_precision);
                y_train = vec(y[train, :]);
                y_pred_train = vec(MLJ.predict(mach, rows = train));
                r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
                rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
                model_perform = [k tuning_param r2_test r2_train rmse_test rmse_train];
                if (tuning_param == tuning_param_rng[1]) & (k == 1)
                    CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, tuning_param_label, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]));
                else
                    CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, tuning_param_label, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true);
                end
                model_perform_mat = vcat(model_perform_mat, model_perform)
                model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[5], x[6]), rev = false);
                model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[3], x[4]), rev = true);
                model_perform_mat = model_perform_mat[1, :]'
                model_perform_df = DataFrame(model_perform_mat, [:iter, tuning_param_label, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]);
            end
        end
    end
    return model_perform_df :: DataFrame
end
