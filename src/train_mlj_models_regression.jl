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
    tuning_param_low :: Number,
    tuning_param_high :: Number,
    tuning_param_step :: Number,
    cv_strategy :: Any = nothing,
    niter :: Int64 = 500,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2)
    model_perform = Array{Float64}(undef, 0, 7);
    model_perform_mat = Array{Float64}(undef, 0, 7);
    model_perform_df = DataFrame();
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    for tuning_param in tuning_param_low : tuning_param_step : tuning_param_high
        tuning_param_name = tuning_param;
        for i in 1:niter
            if isnothing(cv_strategy)
                train = eachindex(y);
                mach = MLJ.machine(mlj_model, x, y);
                MLJ.fit!(mach, rows = train, verbosity = 0);
                y_train = vec(y[train, :]);
                y_pred_train = vec(MLJ.predict(mach, rows = train));
                r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
                rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
                MLJ.save(save_trained_model_at * "/trained_model.jlso", mach, compression = :none);
                model_perform = [r2_train rmse_train];
                model_perform_df = DataFrame(model_perform[1, :]', [:r_squared_train, :rmse_train]);
                CSV.write(save_trained_model_at * "/model_training_records.csv", model_perform_df, append = true);
            else
                if typeof(cv_strategy) == Tuple{Vector{Int64}, Vector{Int64}}
                    kiter = 1
                else
                    kiter = size(cv_strategy)[1]
                end
                for k in 1:kiter
                    if typeof(cv_strategy) == Tuple{Vector{Int64}, Vector{Int64}}
                        train, test = cv_strategy;
                    else
                        train, test = cv_strategy[kiter, ];
                    end
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
                    model_perform = [i tuning_param r2_test r2_train rmse_test rmse_train];
                    CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, tuning_param_label, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true);
                    r2_flag = ifelse.(isnan.(model_perform[:, 3]) .& isnan.(model_perform[:, 4]), -3,
                    ifelse.(isnan.(model_perform[:, 3]), -2,
                    ifelse.(isnan.(model_perform[:, 4]), -1,
                    ifelse.(model_perform[:, 3] .< model_perform[:, 4], 1, 0))));
                    model_perform = [model_perform r2_flag]
                    model_perform_mat = vcat(model_perform_mat, model_perform)
                    if i > 1
                        model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[5], x[6]), rev = false);
                        model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[3], x[4]), rev = true);
                        model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> x[7], rev = true);
                    end
                    model_perform_mat = model_perform_mat[1, :]'
                    model_perform_df = DataFrame(model_perform_mat, [:iter, tuning_param_label, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train, :r2_flag]);
                end
            end
        end
    end
    return model_perform_df :: DataFrame
end
