# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
# Loss function for Flux models
function loss(flux_model, loss_init, x, y)
    y_pred = vec(flux_model(x))
    return loss_init(y_pred, vec(y))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, loss, loss_init, data, optimizer)
    ps = Flux.params(flux_model)
    for d in data
        train_loss, back = Zygote.pullback(() -> loss(flux_model, loss_init, d...), ps)
        if isnan(train_loss) == true
            train_loss = 0
        else
            train_loss = train_loss
        end
        gs = back(one(train_loss))
        Flux.update!(optimizer, ps, gs)
    end
 end
