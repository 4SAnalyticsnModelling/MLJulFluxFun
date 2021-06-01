using Flux;
using Flux.Zygote;
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
    niter :: Int64 = 500,
    n_epochs :: Int64 = 200,
    nobs_per_batch :: Int64 = 1,
    r_squared_precision :: Int64 = 3,
    rmse_precision :: Int64 = 2,
    optimizer = Flux.Optimise.ADAM())
    model_perform = Array{Float64}(undef, 0, 6);
    model_perform_mat = Array{Float64}(undef, 0, 6);
    model_perform_df = DataFrame();
    rm(save_trained_model_at, force = true, recursive = true);
    mkdir(save_trained_model_at);
    for i in 1:niter
        if isnothing(cv_strategy)
            train = eachindex(y);
            x_train = Matrix(x[train, :])';
            y_train = vec(y[train, :]);
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
            y_train = vec(y_train);
            y_pred_train = vec(flux_model(x_train));
            r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
            rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
            weights = Flux.params(Flux.cpu(flux_model));
            BSON.@save(save_trained_model_at * "/trained_model.bson", weights);
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
                r2_test = round((Statistics.cor(y_test, y_pred))^2, digits = r_squared_precision);
                rmse_test = round(rmse(y_test, y_pred), digits = rmse_precision);
                y_train = vec(y_train);
                y_pred_train = vec(flux_model(x_train));
                r2_train = round((Statistics.cor(y_train, y_pred_train))^2, digits = r_squared_precision);
                rmse_train = round(rmse(y_train, y_pred_train), digits = rmse_precision);
                model_perform = [i r2_test r2_train rmse_test rmse_train];
                CSV.write(save_trained_model_at * "/model_training_records.csv", DataFrame(model_perform, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train]), append = true);
                r2_flag = ifelse.(isnan.(model_perform[:, 2]) .& isnan.(model_perform[:, 3]), -3,
                ifelse.(isnan.(model_perform[:, 2]), -2,
                ifelse.(isnan.(model_perform[:, 3]), -1,
                ifelse.(model_perform[:, 2] .< model_perform[:, 3], 1, 0))));
                model_perform = [model_perform r2_flag]
                model_perform_mat = vcat(model_perform_mat, model_perform)
                if i > 1
                    model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[4], x[5]), rev = false);
                    model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> (x[2], x[3]), rev = true);
                    model_perform_mat = sortslices(model_perform_mat, dims = 1, by = x -> x[6], rev = true);
                end
                model_perform_mat = model_perform_mat[1, :]'
                model_perform_df = DataFrame(model_perform_mat, [:iter, :r_squared_test, :r_squared_train, :rmse_test, :rmse_train, :r2_flag]);
                end
            end
        end
    return model_perform_df :: DataFrame
end
