"""The Scan object. - The Parent Of ALL Scans"""
import datetime

import numpy as np
from mag_utils.mag_utils.saver import save_as_csv

from .._consts import Sequence
from ..functional.time import compute_sampling_rate
from ..saver import save_as_h5


class Scan:
    """Object containing the data of a scan."""

    def __init__(self,
                 file_name: str,
                 b: Sequence,
                 time: Sequence,
                 date: datetime.date = None,
                 sampling_rate: float = None):
        """
        Create an instance of Scan object.

        Args:
            file_name: Path to the scan file.
            b: magnetic field [nT].
            time: time.
            date: date.
            sampling_rate: sensor sampling rate [Hz]
        """
        if b is None or len(b) == 0:
            raise ValueError('Trying to create a scan with no data.')
        if time is not None and not isinstance(time[0], datetime.time):
            raise ValueError(f'Time should contain datetime.time objects. Got {type(time[0])} instead.')

        self.file_name = file_name
        self.b = np.asarray(b)
        self.time = np.asarray(time)
        self.date = date
        self.sampling_rate = sampling_rate

        if sampling_rate is None and time is not None:
            self.sampling_rate = compute_sampling_rate(time)

    @staticmethod
    def _item_indexer(item, key):
        """
        Item indexer for __getitem__ method.

        Args:
            item: The item.
            key: The key to index by if possible.

        Returns:
            The indexed item.
        """
        if isinstance(item, np.ndarray):
            return item[key]

        return item

    @staticmethod
    def _items_appender(first_item, second_item):
        """
        Items appender for __append__ method.

        Args:
            first_item: The first item.
            second_item: The second item.

        Returns:
            The appended items by how it should be appended.
        """
        # this is not elif for prospector reasons
        if isinstance(first_item, np.ndarray):
            return np.concatenate([first_item, second_item])

        if isinstance(first_item, bool):
            return first_item and second_item

        if first_item is not None and second_item is not None:
            return first_item + second_item

        return first_item if first_item is not None else second_item

    def __getitem__(self, key) -> 'Scan':
        """
        Create a new sliced scan.

        Args:
            key: The key is whats inside the squared-brackets ([]).
                 When you slice (Scan[start:stop]) the start:stop sent as 'slice' object.
                 The key can be mask too, that means it can be list/ndarray of boolean values.

        Returns:
            New Scan - Sliced.
        """
        self_attrs = {attr_name: self._item_indexer(attr_value, key) for attr_name, attr_value in self.__dict__.items()
                      if attr_name[0] != "_"}
        self_attrs["sampling_rate"] = None
        scan = self.__class__(**self_attrs)

        return scan

    def __eq__(self, other) -> bool:
        """
        Check whether the two objects are equals.

        Args:
            other: Scan object.

        Returns:
            boolean that indicate whether the objects are the same.
        """
        if self.__class__ == other.__class__:
            for self_attr, other_attr in zip(self.__dict__.values(), other.__dict__.values()):
                if isinstance(self_attr, np.ndarray) and self_attr.shape == other_attr.shape:
                    if not (self_attr == other_attr).all():
                        return False
                elif not self_attr == other_attr:
                    return False
        else:
            raise TypeError(f"Can't compare object of type {other.__class__} to {self.__class__}")

        return True

    def __len__(self):
        """Return the length of the scan."""
        return len(self.b)

    def append(self, other: 'Scan'):
        """
        Append BaseScans.

        Args:
            other: the other Scan to append to this one.

        Returns:
            A new appended Scan.
        """
        self_attrs = {attr_name: self._items_appender(first_attr_value, second_attr_value) for
                      (attr_name, first_attr_value), second_attr_value in
                      zip(self.__dict__.items(), other.__dict__.values()) if attr_name[0] != "_"}

        return self.__class__(**self_attrs)

    def save(self, path: str):
        """
        Save the object data in a file according to the file type.

        Supported file types: h5

        Args:
            path: The output file path.
        """
        if path.lower().endswith(".h5") or path.lower().endswith(".hdf5"):
            save_as_h5.save(output_path=path, scan=self)
        if path.lower().endswith(".txt") or path.lower().endswith(".csv"):
            save_as_csv.save(output_path=path, scan=self)
