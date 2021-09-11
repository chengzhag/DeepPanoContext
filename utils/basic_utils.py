import numpy as np
import pickle
import json
import torch
import pickle5
from external.ldif.representation.structured_implicit_function import StructuredImplicit


def read_pkl(pkl_file, protocol=4):
    with open(pkl_file, 'rb') as file:
        if protocol == 4:
            data = pickle.load(file)
        elif protocol == 5:
            data = pickle5.load(file)
        else:
            raise NotImplementedError()
    return data


def write_pkl(obj, pkl_file):
    with open(pkl_file, 'wb') as file:
        pickle.dump(obj, file)


def read_json(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    return json_data


def write_json(obj, json_file):
    with open(json_file, 'w') as file:
        json.dump(obj, file, indent=4)


def get_flag(flags, k):
    if isinstance(flags, dict):
        flag = flags.get(k, None)
    elif isinstance(flags, (list, tuple, set)):
        flag = k in flags
    else:
        flag = flags
    return flag


def list_of_dict_to_dict_of_array(old_lis, force_list=None, force_cat=None, stop_recursion=None, to_tensor=None):
    """
    From a list of dict to a dict of array (or list)
    """
    if isinstance(old_lis, list):
        if not old_lis:
            return {}
        elif not isinstance(old_lis[0], dict):
            return old_lis
    else:
        return old_lis

    new_dic = {}
    if len(old_lis) == 0:
        return new_dic

    key_type = {}
    for d in old_lis:
        key_type.update({k: type(v) for k, v in d.items()})

    for k, t in key_type.items():
        if_force_list = get_flag(force_list, k)
        if_force_cat = get_flag(force_cat, k)
        if_stop_recursion = get_flag(stop_recursion, k)
        if_to_tensor = get_flag(to_tensor, k)

        new_value = [dic[k] for dic in old_lis if k in dic]

        if t == list and if_force_cat == True:
            new_value = [v for l in new_value for v in l]
            if_force_cat = False

        if if_stop_recursion != True:
            new_value = list_of_dict_to_dict_of_array(
                new_value, if_force_list, if_force_cat, if_stop_recursion, if_to_tensor)

        if not (isinstance(new_value, dict) or if_force_list == True):
            if t == torch.Tensor:
                new_value = torch.cat(new_value) if if_force_cat else torch.stack(new_value)
            elif t == list and if_force_cat:
                new_value = [v for l in new_value for v in l]
            elif t == StructuredImplicit:
                new_value = StructuredImplicit.cat(new_value)
            elif t not in (str, dict):
                new_value = np.concatenate(new_value) if if_force_cat else np.stack(new_value)

        if if_to_tensor == True:
            new_value = recursively_to(new_value, dtype='tensor')
        new_dic[k] = new_value

    return new_dic


def dict_of_array_to_list_of_dict(old_dic, split=None, keep_dim=False):
    """
    From a dict of array (or list) to a list of dict
    """
    new_list = []
    if len(old_dic) > 0:
        for key, value in old_dic.items():
            if isinstance(value, dict):
                value = dict_of_array_to_list_of_dict(value)
                if isinstance(split, dict) and split and key in split:
                    value = [value[start:end] for start, end in split[key]]
            if new_list:
                assert len(new_list) == len(value)
            for i_obj in range(len(value)):
                obj = value[i_obj:i_obj+1] if keep_dim else value[i_obj]
                if i_obj >= len(new_list):
                    new_list.append({})
                obj = recursively_to(obj, dtype='numpy')
                if isinstance(obj, np.ndarray) and obj.size == 1:
                    if obj.dtype in (np.float32, np.float64):
                        obj = float(obj)
                    elif obj.dtype == np.bool:
                        obj = bool(obj)
                    else:
                        obj = int(obj)
                new_list[i_obj][key] = obj
        if isinstance(split, torch.Tensor):
            new_list = [new_list[start:end] for start, end in split]
    return new_list


def recursively_to(data, dtype=None, device=None):
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_data[k] = recursively_to(v, dtype, device)
    elif isinstance(data, (list, tuple, set)):
        new_data = []
        for v in data:
            new_data.append(recursively_to(v, dtype, device))
        new_data = type(data)(new_data)
    elif isinstance(data, torch.Tensor):
        if dtype in ('numpy', 'list'):
            new_data = data.detach().cpu().numpy()
            if dtype == 'list':
                new_data = new_data.tolist()
        elif dtype == 'cuda':
            new_data = data.cuda()
        else:
            new_data = data
        if device is not None:
            new_data = new_data.to(device)
    elif isinstance(data, np.ndarray):
        if dtype == 'list':
            new_data = data.tolist()
        elif dtype in ('tensor', 'cuda'):
            new_data = torch.from_numpy(data)
            if new_data.type() == 'torch.DoubleTensor':
                new_data = new_data.float()
            if dtype == 'cuda':
                new_data = new_data.cuda()
            if device is not None:
                new_data = new_data.to(device)
        else:
            new_data = data
    else:
        new_data = data
    return new_data


def recursively_update(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            recursively_update(dict1[k], v)
        else:
            dict1[k] = v


def get_any_array(dic):
    if isinstance(dic, dict):
        vs = dic.values()
    elif isinstance(dic, (list, tuple, set)):
        vs = dic
    else:
        return dic

    for v in vs:
        t = get_any_array(v)
        if isinstance(t, (np.ndarray, torch.Tensor)):
            return t

    return dic


def recursively_ignore(dic, ignore_keys):
    if not isinstance(dic, dict):
        return dic
    if isinstance(ignore_keys, dict):
        return {k: recursively_ignore(v, ignore_keys.get(k, {})) for k, v in dic.items() if ignore_keys.get(k) != True}
    else:
        return {k: recursively_ignore(v, {}) for k, v in dic.items() if k not in ignore_keys}
