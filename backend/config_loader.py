"""Load and validate drone configuration from YAML."""

import os
import yaml

DEFAULTS = {
    "vehicle": {
        "type": "copter",
        "frame": "quad",
    },
    "simulation": {
        "ardupilot_dir": "~/ardupilot",
        "home_lat": 50.450001,
        "home_lon": 30.523333,
        "home_alt": 180,
        "home_heading": 0,
        "speed_up": 1,
        "mavlink_port": 14550,
    },
    "visualization": {
        "ground_size": 500,
        "camera_follow": True,
        "camera_distance": 20,
        "show_trail": True,
        "trail_length": 200,
        "drone_model": {
            "body_color": "#333333",
            "prop_color": "#22aa44",
        },
    },
}

VEHICLE_MAP = {
    "copter": "ArduCopter",
    "plane": "ArduPlane",
    "rover": "Rover",
    "sub": "ArduSub",
}

FRAME_ARM_COUNT = {
    "quad": 4,
    "hexa": 6,
    "octa": 8,
    "tri": 3,
    "y6": 6,
}


def _deep_merge(defaults, overrides):
    """Merge overrides into defaults, returning a new dict."""
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path=None):
    """Load drone config from YAML, applying defaults for missing keys."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "drone.yaml"
        )

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}

    config = _deep_merge(DEFAULTS, user_config)

    # Expand ~ in ardupilot_dir
    config["simulation"]["ardupilot_dir"] = os.path.expanduser(
        config["simulation"]["ardupilot_dir"]
    )

    # Add derived fields
    vehicle_type = config["vehicle"]["type"]
    config["vehicle"]["ardupilot_vehicle"] = VEHICLE_MAP.get(vehicle_type, "ArduCopter")
    config["vehicle"]["arm_count"] = FRAME_ARM_COUNT.get(
        config["vehicle"]["frame"], 4
    )

    return config


def validate_config(config):
    """Validate that ArduPilot is installed and config values are sane."""
    errors = []

    ardupilot_dir = config["simulation"]["ardupilot_dir"]
    if not os.path.isdir(ardupilot_dir):
        errors.append(f"ArduPilot directory not found: {ardupilot_dir}")
    else:
        sim_vehicle = os.path.join(ardupilot_dir, "Tools", "autotest", "sim_vehicle.py")
        if not os.path.isfile(sim_vehicle):
            errors.append(f"sim_vehicle.py not found at: {sim_vehicle}")

    if config["vehicle"]["type"] not in VEHICLE_MAP:
        errors.append(
            f"Unknown vehicle type: {config['vehicle']['type']}. "
            f"Must be one of: {list(VEHICLE_MAP.keys())}"
        )

    sim = config["simulation"]
    if not (-90 <= sim["home_lat"] <= 90):
        errors.append(f"Invalid home_lat: {sim['home_lat']}")
    if not (-180 <= sim["home_lon"] <= 180):
        errors.append(f"Invalid home_lon: {sim['home_lon']}")

    return errors
