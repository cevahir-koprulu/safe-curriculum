import copy
import collections.abc

def update_params(d, u):
    """Update a dictionary with multiple levels

    Args:
        d (TYPE): The dictionary to update
        u (TYPE): The dictionary that contains keys/values to add to d

    Returns:
        TYPE: A new dictionary d
    """
    d = d.copy() # copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_params(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def verbose_print(verbose, s):
    if verbose:
        print(s)

def verbose_input(verbose, s):
    if verbose:
        input(s)