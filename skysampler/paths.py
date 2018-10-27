"""

"""

import os
import yaml


def get_poject_path(user_cfg):
    """Loads the absolute path to the cluster pipeline"""
    path = os.path.expanduser('~') + '/' + user_cfg
    project_path = None
    if os.path.isfile(path):
        with open(path)as file:
            cfg = yaml.safe_load(file)
            project_path = cfg["project_path"]
    return project_path


user_project_file = '.skysampler.yaml'
project_path = get_poject_path(user_project_file)
default_logger_path = project_path + "logs/"





