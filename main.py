
import argparse
from easydict import EasyDict

from agents.naburangi_v1_agent import NaburangiV1Agent
from utils.config import get_config


def main(config):
    """
    Args:
        config (Dict or EasyDict)
    """
    if isinstance(config, EasyDict) is False:
        config = EasyDict(config)

    if config.agent == 'naburangi_v1':
        agent = NaburangiV1Avent(config)

    agent.run()
    agent.finalize()


if __name__ == '__main__':
    # parse the path of the yaml or json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_yaml_or_json_file',
        default='None',
        help='The Configuration file in yaml or json format')
    args = arg_parser.parse_args()

    config = get_config(args.config)

    main(config)
