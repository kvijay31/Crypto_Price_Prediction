from sklearn.metrics import mean_squared_error, mean_absolute_error

def fit_predict(model, X_fit=None, y_fit=None, X_validate=None, y_validate=None, store=None, norm=False):
    if X_fit is None:
        X_fit = X_train.copy()
    if y_fit is None:
        y_fit = y_train.copy()
    if X_validate is None:
        X_validate = X_test.copy()
    if y_validate is None:
        y_validate = y_test.copy()
    if hasattr(model, "set_params"):
        try:
            model.set_params(**{"random_state": 42})
        except:
            pass
    model.fit(X_fit.values, y_fit.values)
    predictions = model.predict(X_validate.values)
    predictions_train = model.predict(X_fit.values)
    if norm:
        y_validate = y_validate * test["std"] + test["mean"]
        predictions = predictions * test["std"] + test["mean"]
    mae = mean_absolute_error(y_validate, predictions)
    mse = mean_squared_error(y_validate, predictions)
    mae_train = mean_absolute_error(y_fit, predictions_train)
    mse_train = mean_squared_error(y_fit, predictions_train)
    scores = {"mae": mae, "mse": mse, "train_mae": mae_train, "train_mse": mse_train}
    if store is not None:
        if hasattr(model, "get_params"):
            params = model.get_params()
        else:
            params = model.__dict__
        store.add(model, scores, predictions, y_validate.to_list(), params)
    return model, predictions, scores

