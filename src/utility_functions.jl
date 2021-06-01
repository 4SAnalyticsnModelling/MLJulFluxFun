# Utility and wrapper functions to be used in training models
using Flux;
using Flux.Zygote;
# Root mean squared error
function rmse(obs :: Vector, pred :: Vector)
    return sqrt(sum((obs .- pred).^2)/length(obs))
end
# Loss function for Flux models
function loss(flux_model, x, y)
    return sqrt(sum((y .- vec(flux_model(x))).^2)/length(y))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, ps, data, optimizer)
    ps = Zygote.Params(ps)
    for d in data
        train_loss, back = Zygote.pullback(() -> loss(flux_model, d...), ps)
        gs = back(one(train_loss))
        Flux.Optimise.update!(optimizer, ps, gs)
    end
end
