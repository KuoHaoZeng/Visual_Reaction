"""config utilities for yml file."""
import os
from ruamel import yaml

class AttrDict(dict):
    """Dict as attribute trick.

    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return.

        """
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables.

        """
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    """Config with yaml file.

    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.

    Args:
        filename(str): File name.

    Examples:

        yaml file ``model.yml``::

            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'

        Usage in .py:

        >>> from neuralgym import Config
        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet

    """

    def __init__(self, filename=None):
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        self.all_keys = []
        self.org_dict = dict(cfg_dict)
        cfg_dict = self.replace_variable(cfg_dict)
        cfg_dict = self.replace_variable(cfg_dict)
        super(Config, self).__init__(cfg_dict)

    def replace_variable(self, cfg_dict):
        output_dict = {}
        for k, v in cfg_dict.items():
            self.all_keys.append(k)
            if isinstance(v, dict):
                v = self.replace_variable(v)
            if isinstance(v, str):
                if "{{" in v and "}}" in v:
                    num = v.count("{{")
                    for _ in range(num):
                        target_key = v.split("{{")[1].split("}}")[0]
                        if target_key in self.all_keys:
                            pos_remain = "}}".join(v.split("}}")[1:])
                            pre_remain = v.split("{{")[0]
                            v = "{}{}{}".format(pre_remain, self.org_dict[target_key], pos_remain)
                        else:
                            raise KeyError
            output_dict[k] = v
        return output_dict
