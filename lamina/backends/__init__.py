"""
lamina.backends
===============
Pluggable storage backends for serialising ``InternalsRecord`` lists.

Each backend implements the ``Backend`` ABC: a ``write`` method that
persists records and a ``read`` classmethod that loads them back.

Available backends
------------------
``FilesystemBackend`` (``lamina.backends.filesystem``)
    Writes one ``.npz`` file per record plus a ``metadata.jsonl`` index.
    No additional dependencies — works with numpy only.

``HFDatasetBackend`` (``lamina.backends.hf_dataset``)
    Converts records to a HuggingFace ``datasets.Dataset`` for Arrow
    serialisation and easy ``push_to_hub()``.
    Requires: ``pip install lamina[backends]``

Planned backends
----------------
``RedisBackend``    — store runs in Redis (key: run_id, value: msgpack)
``MongoBackend``    — store runs as BSON documents in MongoDB

Convenience wrappers
--------------------
The top-level ``lamina.dump()`` and ``lamina.load()`` functions delegate
to ``FilesystemBackend`` for backward compatibility.
"""
from .base import Backend
from .filesystem import FilesystemBackend, dump, load
from .hf_dataset import to_hf_dataset

__all__ = [
    "Backend",
    "FilesystemBackend",
    "dump",
    "load",
    "to_hf_dataset",
]
