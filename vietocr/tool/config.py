import os

try:
    from ruamel.yaml import YAML

    _yaml = YAML(typ="safe")
    _yaml.allow_unicode = True

    def _load_yaml(path):
        with open(path, encoding="utf-8") as f:
            return _yaml.load(f)

    def _dump_yaml(data, stream):
        _yaml.dump(data, stream)
except Exception:
    import yaml as _pyyaml

    def _load_yaml(path):
        with open(path, encoding="utf-8") as f:
            return _pyyaml.safe_load(f)

    def _dump_yaml(data, stream):
        _pyyaml.safe_dump(
            data,
            stream,
            default_flow_style=False,
            allow_unicode=True,
        )

def load_config(config_file):
    return _load_yaml(config_file)

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname, base_file=None):
        if base_file is None:
            base_file = os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                "base.yml",
            )
        base_config = load_config(os.path.abspath(base_file))
        config = load_config(os.path.abspath(fname))
        base_config.update(config)
        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name, base_file=None):
        if not name.endswith(".yml"):
            name = f"{name}.yml"
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            name,
        )
        return Cfg.load_config_from_file(cfg_path, base_file=base_file)


    def save(self, fname):
        with open(fname, 'w') as outfile:
            _dump_yaml(dict(self), outfile)
