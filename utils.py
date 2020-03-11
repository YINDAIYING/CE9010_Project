import torch

#three functions for federated average
def add_model(dst_model, src_model, dst_no_data, src_no_data):
    """Add the parameters of two models.
        Args:
            dst_model (torch.nn.Module): the model to which the src_model will be added.
            src_model (torch.nn.Module): the model to be added to dst_model.
        Returns:
            torch.nn.Module: the resulting model of the addition.
        """
    if (dst_model==None):
        return src_model
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models,data_num):
    """Calculate the federated average of a list of models.
    Args:
        models: the dictionary of models of which the federated average is calculated.
    Returns:
        torch.nn.Module: the module with averaged parameters.
    """

    total_no_data=0
    model=None
    for i in models.keys():
        model = add_model(model, models[i],total_no_data,data_num[i])
        model = scale_model(model, 1.0 / (total_no_data+data_num[i]))
        total_no_data=total_no_data+data_num[i]
    return model
