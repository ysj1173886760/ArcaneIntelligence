import json
from typing import Any, Dict

class GlobalConfig:
    _config: Dict[str, Any] = {}
    @classmethod
    def init_config(cls, config_json_path: str):
      with open(config_json_path, 'r') as config_file:
        cls._config = json.load(config_file)

    @classmethod
    def get_config(cls, config_name: str) -> Any:
      parts = config_name.split('.')
      value = cls._config
      for part in parts:
          if isinstance(value, dict) and part in value:
              value = value[part]
          else:
              return None
      return value
  