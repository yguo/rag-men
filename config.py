import os
import configparser
from typing import Any

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        # Load the template config
        self.config.read('config.template.ini')
        
        # Override with local config if it exists
        local_config_path = 'config.local.ini'
        if os.path.exists(local_config_path):
            self.config.read(local_config_path)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        return self.config.get(section, key, fallback=fallback)

# Create a global instance of the Config class
config = Config()

