"""
Abstract base class for lamina storage backends.

A backend is responsible for persisting a list of ``InternalsRecord``
objects and reading them back.  The interface is intentionally minimal so
that implementations for very different storage systems (files, Redis,
MongoDB, cloud object stores) can share the same calling convention.

Implementing a new backend
--------------------------
Subclass ``Backend`` and implement ``write`` and ``read``::

    from lamina.backends.base import Backend
    from lamina.applications.dataset import InternalsRecord

    class MyBackend(Backend):
        def __init__(self, connection_string: str) -> None:
            self._conn = connect(connection_string)

        def write(self, records, **kwargs) -> None:
            for record in records:
                self._conn.put(record.run.run_id, serialise(record))

        @classmethod
        def read(cls, source, **kwargs):
            conn = connect(source)
            arrays_list = [deserialise(v) for v in conn.scan()]
            meta_list   = [extract_meta(v) for v in conn.scan()]
            return arrays_list, meta_list

Planned concrete backends
--------------------------
``RedisBackend``
    Store each run as a msgpack-serialised document keyed by ``run_id``.
    Suitable for real-time streaming scenarios where many workers push
    internals concurrently.

``MongoBackend``
    Store each run as a BSON document.  Allows rich querying by
    ``properties`` fields without loading array data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class Backend(ABC):
    """
    Abstract base for lamina storage backends.

    Subclasses must implement :meth:`write` and :meth:`read`.
    """

    @abstractmethod
    def write(
        self,
        records: List[Any],   # List[InternalsRecord]
        **kwargs: Any,
    ) -> None:
        """
        Persist *records* to storage.

        Parameters
        ----------
        records : list[InternalsRecord]
        **kwargs
            Backend-specific options (compression level, collection name, …).
        """

    @classmethod
    @abstractmethod
    def read(
        cls,
        source: Any,
        **kwargs: Any,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load previously written records.

        Returns
        -------
        arrays_list : list[dict[str, np.ndarray]]
            One dict of arrays per record.
        metadata_list : list[dict]
            One dict of scalar metadata per record.
        """
