import collections
import numpy as np
import scipy


def inspect_params(params,
                   expected,
                   fail_if_extra=True,
                   fail_if_missing=True):
    """Inspects whether the params are consistent with the expected keys."""
    params_flat = _flatten_dict(params)
    expected_flat = _flatten_dict(expected)

    # 차집합 key 구하기
    missing_keys = expected_flat.keys() - params_flat.keys()
    extra_keys = params_flat.keys() - expected_flat.keys()

    # Adds back empty dict explicitly, to support layers without weights.
    # Context: FLAX ignores empty dict during serialization.
    empty_keys = set()
    for k in missing_keys:
        if isinstance(expected_flat[k], dict) and not expected_flat[k]:
            params[k] = {}
            empty_keys.add(k)
    missing_keys -= empty_keys

    if empty_keys:
        logging.warning('Inspect recovered empty keys:\n%s', empty_keys)
    if missing_keys:
        logging.info('Inspect missing keys:\n%s', missing_keys)
    if extra_keys:
        logging.info('Inspect extra keys:\n%s', extra_keys)

    if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
        raise ValueError(f'Missing params from checkpoint: {missing_keys}.\n'
                        f'Extra params in checkpoint: {extra_keys}.\n'
                        f'Restored params from checkpoint: {params_flat.keys()}.\n'
                        f'Expected params from code: {expected_flat.keys()}.')
    return params