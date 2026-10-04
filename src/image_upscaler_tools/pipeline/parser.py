import ast
import re
from typing import Any, Dict, List, Tuple


def parse_step_string(step_str: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parse a string of pipeline steps separated by '>' or '->' or '|'.
    Supports function-call syntax for arguments, e.g.:
      "fast_nlm > span"
      "fast_nlm(h=3, h_color=4) > hat"
      "adaptive_median(s_max=5) -> ultrasharp"
      "median | bilateral(d=5) | span"
    
    Returns
    -------
    List of (step_name, params_dict) tuples.
    """
    raw_steps = re.split(r"\s*(?:->|>|\|)\s*", step_str.strip())
    parsed_steps = []

    for item in raw_steps:
        item = item.strip()
        if not item:
            continue

        match = re.match(r"^([a-zA-Z0-9_]+)(?:\((.*)\))?$", item)
        if not match:
            raise ValueError(f"Invalid pipeline step syntax: '{item}'")

        name = match.group(1).lower().strip()
        args_str = match.group(2)
        params = {}

        if args_str:
            # Parse keyword arguments like: h=3, sigma_color=0.03, method="soft"
            # Using ast to safely parse literals
            arg_pairs = [a.strip() for a in args_str.split(",") if a.strip()]
            for pair in arg_pairs:
                if "=" not in pair:
                    raise ValueError(
                        f"Positional parameters not supported in step string '{item}'. "
                        f"Use keyword arguments, e.g., 'fast_nlm(h=3)'."
                    )
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    parsed_val = ast.literal_eval(v)
                except Exception:
                    parsed_val = v
                params[k] = parsed_val

        parsed_steps.append((name, params))

    if not parsed_steps:
        raise ValueError(f"No valid steps found in step string: '{step_str}'")

    return parsed_steps
