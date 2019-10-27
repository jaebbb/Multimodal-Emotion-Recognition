import yaml
from easydict import EasyDict


def get_config(file):
    """Get the config from a yaml file
    Args:
        json_file (str): Path of yaml or json file
    Returns:
        config (EasyDict[str, any]): Config
    """

    with open(file, 'r') as f:
        try:
            config = yaml.load(f, yaml.Loader)
            config = EasyDict(config)
            return config
        except:
            print(
                "INVALID YAML or JSON file format.. Please provide a good yaml or json file")
            exit(-1)
