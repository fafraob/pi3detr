from typing import Optional
from argparse import Namespace
from types import SimpleNamespace
import yaml


def load_yaml(file_path: str) -> yaml.YAMLObject:
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            yaml.YAMLError("error reading yaml file")


def load_args(config: str, parsed_args: Optional[Namespace] = None) -> SimpleNamespace:
    args = load_yaml(config)
    parsed_args = vars(parsed_args) if parsed_args else {}
    args = args | parsed_args
    args = SimpleNamespace(**args)
    return args
