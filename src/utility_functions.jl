# Utility and wrapper functions to be used in training and evaluating models
using Flux
using Flux.Zygote
# Loss function for Flux models
function loss(flux_model, loss_init, x, y)
    y_pred = vec(flux_model(x))
    return loss_init(y_pred, vec(y))
end
# Custom training function for Flux models
function my_custom_train!(flux_model, ps, loss, loss_init, data, optimizer)
    ps = Zygote.Params(ps)
    for d in data
        train_loss, back = Zygote.pullback(() -> loss(flux_model, loss_init, d...), ps)
        gs = back(one(train_loss))
        if any(isnan.(gs)) == true
            Flux.stop()
        end
        Flux.update!(optimizer, ps, gs)
    end
 end
