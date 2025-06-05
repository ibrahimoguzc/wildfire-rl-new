import json
from pathlib import Path
from typing import Any


def combine_parameters(
    defaults: dict[str, Any] | str | Path,
    overwrites: dict[str, Any] | str | Path,
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """Creates an updated dict based on defaults and overwrites.

    `defaults` are the total set of values for the initial dictionary
    upon which the `overwrites` are overlaid. `overwrites` can be a
    dictionary of dictionaries for DoE set-up or a simple dictionary of
    key-value pairs to overwrite the `defaults`. If the former is done,
    the output dictionary has key-value pairs where each one is a unique
    overwritten default dict.
    """

    def load_params(params: str | Path) -> dict[str, Any]:
        with open(params) as f:
            param_dict = json.load(f)
        return param_dict

    def nested_update(d, u):
        updated_dict = {}
        defaults = d.copy()
        for k, v in u.items():
            if k not in d and isinstance(v, dict):
                new_d = defaults.copy()
                updated_dict[k] = nested_update(new_d, v)
            else:
                d[k] = v
        if updated_dict:
            if d != defaults:
                updated_dict["updated_defaults"] = d
            return updated_dict
        return d

    parameters = (
        defaults.copy()
        if isinstance(defaults, dict)
        else load_params(defaults)
    )
    if not isinstance(overwrites, dict):
        overwrites = load_params(overwrites)
    doe_dicts = nested_update(parameters, overwrites)
    return doe_dicts
