import pickle
from typing import List, Dict, Optional, Any
from itertools import islice
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.main.configs import global_configs as configs
from src.main.utility.settings_reader import SettingsReader


class Helpers:
    """
    Helper utilities
    """

    @staticmethod
    def createDirectoryIfItDoesNotExist(directory: str):
        """
        Create a directory if it does not exist
        :param directory: The directory to create
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

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

    @staticmethod
    def serialObject(
            data: Any,
            pickle_path: str
    ) -> None:
        """
        Serialize the data to a pickle file
        :param data: Data to serialize
        :param pickle_path: Pickle path
        :return: None
        """
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def deserializeObject(pickle_path: str) -> Any:
        """
        Deserialize the data from a pickle file
        :param pickle_path: Pickle path
        :return: Returns the deserialized data
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def filterDict(
            input_dict: Dict[str, Any],
            filter_list: List[Any]
    ) -> Dict[str, Any]:
        """
        Remove/filter a input dictionary based on a list
        :param input_dict: The input dictionary
        :param filter_list:
        :return: Filtered dictionary
        """
        output_dict = {
            key: input_dict[key]
            for key in input_dict if key not in filter_list
        }
        return output_dict

    @staticmethod
    def getEnumType(
            enum_type: Any,
            enum_type_name: str
    ) -> Any:
        """
        Gets the enum type based on the specified type name
        :param enum_type: Enum type
        :param enum_type_name: Enum name
        :return: Enum value
        """
        enum_dict = enum_type.__dict__
        enum_value = enum_dict.get(enum_type_name.lower())
        if enum_value is None:
            raise ValueError(f"Enum type {enum_type_name} does not exist!")
        return enum_value

    @staticmethod
    def chunklistCollection(
            lst: List[Any],
            chunk_size: int
    ) -> List[List[Any]]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked list
        """
        it = iter(lst)  # Create an iterator
        return [
            list(islice(it, chunk_size))
            for _ in range((len(lst) + chunk_size - 1) // chunk_size)
        ]

    @staticmethod
    def chunkArray(
            lst: List[Any],
            chunk_size: int
    ) -> List[Any]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked array
        """
        array_list = np.array(lst, dtype=object)
        chunked_array = np.array_split(array_list, chunk_size)
        return chunked_array


    @staticmethod
    def text2Boolean(arg):
        """
        Parses a string to a boolean type
        :param arg: Input argument
        :return: Boolean value
        """
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            print("Invalid argument specified, had to set it to default value of 'True'")
            return True



