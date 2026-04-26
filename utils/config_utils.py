import argparse
import os
import re
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        string = file.read()
        dict = yaml.safe_load(string)
    return dict


def _fuse_dict(dic1, dic2):
    """
    if conflict, use the latter
    """
    for key in dic1.keys():
        if key in dic2.keys():
            if isinstance(dic1[key], dict):
                dic1[key].update(dic2[key])
            else:
                dic1[key] = dic2[key]
    return dic1


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def check_dict(dic):
    for k, v in dic.items():
        # check none
        if v == "None":
            dic[k] = None
            continue
        # check float
        v = convert_str_to_number(v)
        if isinstance(v, float) and v == int(v):
            v = int(v)
        dic[k] = v


def convert_str_to_number(v):
    try:
        return float(v)
    except ValueError:
        return v


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if isinstance(v, dict):
            check_dict(v)
        if v is None or v == "None":
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def merge_two_dicts(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict):
            dict1[k].update(v)
        else:
            dict1[k] = v
    return dict1


def _parse_args_and_yaml(parser):
    args = parser.parse_args()
    args_dict = args.__dict__
    configs_path = args_dict["config"]
    base_configs_path = str(Path(configs_path).parent / "base.yaml")
    args_base = OmegaConf.load(base_configs_path)
    args_ = OmegaConf.load(configs_path)
    args = OmegaConf.merge(args_base, args_)  # agrs_ should overwrite args_base
    args.update(args_dict)
    return args


def _parse_args_and_yaml_in_sampling(parser):
    args = parser.parse_args()
    args_dict = args.__dict__
    configs_path = args_dict["config"]
    sampple_configs_path = args_dict["sample_config"]
    base_configs_path = str(Path(configs_path).parent / "base.yaml")

    configs_dict = OmegaConf.load(configs_path)
    base_configs_dict = OmegaConf.load(base_configs_path)
    sample_configs_dict = OmegaConf.load(sampple_configs_path)

    args = OmegaConf.merge(base_configs_dict, configs_dict, sample_configs_dict)
    args.update(args_dict)
    return args


class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# def load_config_from_yaml(file_path):
#     with open(file_path, 'r') as yaml_file:
#         data = yaml.safe_load(yaml_file)
#     return Config(data)


def _expand_paths(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = _expand_paths(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = _expand_paths(item)
    elif isinstance(data, str):
        # 先替换环境变量占位符 ${ENV_VAR}
        if "${" in data:
            pattern = re.compile(r'\$\{([^}]+)\}')
            def replace_env(match):
                env_var = match.group(1)
                env_value = os.environ.get(env_var, match.group(0))
                # 如果环境变量值也包含~，则展开它
                if env_value.startswith("~"):
                    return os.path.expanduser(env_value)
                return env_value
            data = pattern.sub(replace_env, data)
        # 然后展开 ~ 路径
        if data.startswith("~"):
            return os.path.expanduser(data)
    return data


def _find_unresolved_references(data):
    """Find all unresolved ${...} references in the data structure."""
    unresolved = []

    def _search(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                _search(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _search(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            if re.fullmatch(r"\$\{(.+?)\}", obj):
                unresolved.append(f"{path}: {obj}")

    _search(data)
    return unresolved


def load_config_from_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as yaml_file:  # 添加 encoding
        raw_data = yaml.safe_load(yaml_file)

    if raw_data is None:  # 处理空YAML文件的情况
        return Config({})

    resolved_data = raw_data
    max_iterations = 5  # 防止无限循环
    for iteration in range(max_iterations):
        previous_data_str = str(resolved_data)  # 比较字符串形式来检测变化
        resolved_data = resolve_references(
            resolved_data, raw_data
        )  # 始终从原始数据中查找引用
        if str(resolved_data) == previous_data_str:
            break  # 如果没有变化，说明所有可解析的引用都已处理
    else:
        unresolved_refs = _find_unresolved_references(resolved_data)
        print(
            f"Warning: Max iterations ({max_iterations}) reached for reference resolution in {file_path}. "
            f"Unresolved references found: {unresolved_refs}"
        )

    resolved_data = _expand_paths(resolved_data)

    return Config(resolved_data)


def merge_config(cfg: Config, args: argparse.Namespace):
    cfg_dict = cfg.to_dict()
    args_dict = args.__dict__
    for key, value in args_dict.items():
        cfg_dict[key] = value
    return Config(cfg_dict)


def _get_value_by_path(data_root, path_str):
    keys = path_str.split(".")
    current_value = data_root
    for key in keys:
        if isinstance(current_value, dict) and key in current_value:
            current_value = current_value[key]
        else:
            raise KeyError(f"Path '{path_str}' not found or invalid.")
    return current_value


def resolve_references(current_node, root_data):
    if isinstance(current_node, dict):
        resolved_dict = {}
        for key, value in current_node.items():
            resolved_dict[key] = resolve_references(value, root_data)
        return resolved_dict
    elif isinstance(current_node, list):
        return [resolve_references(item, root_data) for item in current_node]
    elif isinstance(current_node, str):
        match = re.fullmatch(r"\$\{(.+?)\}", current_node)
        if match:
            path_str = match.group(1)
            try:
                resolved_value = _get_value_by_path(root_data, path_str)
                #
                if (
                    resolved_value != current_node
                    and isinstance(resolved_value, str)
                    and re.fullmatch(r"\$\{(.+?)\}", resolved_value)
                ):
                    return resolve_references(resolved_value, root_data)
                return resolved_value
            except KeyError as e:
                print(
                    f"Warning: Could not resolve reference '{current_node}'. Error: {e}"
                )
                return current_node
        else:
            return current_node
    else:
        return current_node
