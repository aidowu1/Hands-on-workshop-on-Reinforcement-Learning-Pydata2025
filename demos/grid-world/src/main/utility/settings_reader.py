import json
from typing import Dict, Optional
import os
from pathlib import Path

from src.main.configs import global_configs as configs


class SettingsReader:
    """
    Utility used to read/deserialize the market simulation settings file
    """
    def __init__(
            self,
            settings_file_name: str
    ):
        """
        Constructor
        :param settings_file_name: Setting file name
        """
        current_path = SettingsReader.getPojectRootPath()
        os.chdir(current_path)
        self._settings_folder = configs.SETTINGS_FOLDER
        self._settings_file_path = f"{self._settings_folder}/{settings_file_name}.json"
        self._settings = {}

    @staticmethod
    def getPojectRootPath(project_name: str = configs.PROJECT_ROOT_PATH) -> str:
        """
        Gets the project root path
        :param project_name: The project name
        :return: Returns the project root path
        """
        path = os.path.dirname(os.path.realpath(__file__))
        while not str(path).endswith(str(project_name)):
            path = Path(path).parent
        return path


    def read(self) -> Dict[str, Optional[str]]:
        """
        Reads the settings file
        :return: Settings object
        """
        with open(self._settings_file_path, 'r') as f:
            self._settings = json.load(f)
        return self._settings

    @property
    def file_exists(self) -> bool:
        """
        Checks if the settings file exists
        :return: Status of file existence
        """
        return os.path.exists(self._settings_file_path)