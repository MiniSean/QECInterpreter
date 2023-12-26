# -------------------------------------------
# Functions for import/export and formatting hdf5.
# -------------------------------------------
import os
import warnings
import h5py
from h5py import File, Dataset
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


class SpecType(Enum):
    DATASET = "dset"
    ATTRIBUTE = "attr"
    ALL_ATTRIBUTES = "attr:all"
    GROUP = "group"


@dataclass
class ExtractionSpec:
    path: str
    spec_type: SpecType
    attr_name: Optional[str] = None  # Only used if spec_type is ATTRIBUTE
    key: Optional[str] = None  # Optional custom key for the extracted data


class IDataExtractor(ABC):
    @classmethod
    @abstractmethod
    def extract_dataset(cls, file, path: str):
        """Extracts a dataset from the file."""
        pass

    @classmethod
    @abstractmethod
    def extract_attribute(cls, file, path: str, attr_name: str):
        """Extracts a single attribute from the file."""
        pass

    @classmethod
    @abstractmethod
    def extract_all_attributes(cls, file, path: str):
        """Extracts all attributes from the file."""
        pass

    @classmethod
    @abstractmethod
    def extract_group(cls, file, path: str):
        """Recursively extracts all data from a group in the file."""
        pass


class HDF5DataExtractor(IDataExtractor):
    @classmethod
    def extract_dataset(cls, file, path: str):
        return file[path][()]

    @classmethod
    def extract_attribute(cls, file, path: str, attr_name: str):
        return file[path].attrs[attr_name]

    @classmethod
    def extract_all_attributes(cls, file, path: str):
        return dict(file[path].attrs)

    @classmethod
    def extract_group(cls, file, path: str):
        result = {}
        group = file[path]
        for key in group:
            if isinstance(group[key], h5py.Dataset):
                result[key] = group[key][()]
            elif isinstance(group[key], h5py.Group):
                result[key] = cls.extract_group(file, f"{path}/{key}")
        return result


def extract_data(file_path: str, specs: List[ExtractionSpec], extractor: IDataExtractor = HDF5DataExtractor) -> Dict[str, Any]:
    """Extracts data from a file based on a list of ExtractionSpec objects using the provided extractor."""
    extracted_data = {}
    with h5py.File(file_path, 'r') as file:
        for spec in specs:
            data_key = spec.key if spec.key else spec.path
            try:
                if spec.spec_type == SpecType.DATASET:
                    extracted_data[data_key] = extractor.extract_dataset(file, spec.path)
                elif spec.spec_type == SpecType.ATTRIBUTE:
                    extracted_data[data_key] = extractor.extract_attribute(file, spec.path, spec.attr_name)
                elif spec.spec_type == SpecType.ALL_ATTRIBUTES:
                    extracted_data[data_key] = extractor.extract_all_attributes(file, spec.path)
                elif spec.spec_type == SpecType.GROUP:
                    extracted_data[data_key] = extractor.extract_group(file, spec.path)
                else:
                    print(f"Unknown specification type '{spec.spec_type}' for path '{spec.path}'")
            except KeyError as e:
                print(f"KeyError for path '{spec.path}': {e}")
            except Exception as e:
                print(f"An error occurred while processing path '{spec.path}': {e}")

    return extracted_data
