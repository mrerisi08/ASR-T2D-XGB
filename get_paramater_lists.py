import itertools


def get_param_list(params):
    """
    Takes a dictionary containing either lists or single values.
    For each key it assigns each of its values with each of every other keys' values.
    Converts {"a":[1, 2], "b":[3, 4]} to [{"a":1, "b":3}, {"a":1, "b":4}, {"a":2, "b":3}, ...]
    """
    products = list(itertools.product(*[params[k] for k in params if isinstance(params[k], list)]))
    list_names = [k for k in params if isinstance(params[k], list)]
    not_list_names = [k for k in params if not isinstance(params[k], list)]
    out_list = []
    for param_set in products:
        dict = {}
        for dex in range(len(param_set)):
            dict[list_names[dex]] = param_set[dex]
        for name in not_list_names:
            dict[name] = params[name]
        out_list.append(dict)
    return out_list
