"""Common dataset-management package for clean dense/dynamic runners."""

from .backend_factory import PreparedBackends, build_backends_for_profile
from .hub import DatasetHubAdapters, prepare_dataset_bundle
from .pool_registry import PoolRegistry
from .prepared import PreparedDatasetBundle
from .profiles import DatasetProfile, get_dataset_profile, prepare_dataset_bundle_for_profile
from .runtime_backend import PreindexedRuntimeDatasetBackend
from .spec import DatasetSpec

__all__ = [
    "DatasetHubAdapters",
    "DatasetProfile",
    "DatasetSpec",
    "PoolRegistry",
    "PreparedBackends",
    "PreparedDatasetBundle",
    "PreindexedRuntimeDatasetBackend",
    "build_backends_for_profile",
    "get_dataset_profile",
    "prepare_dataset_bundle",
    "prepare_dataset_bundle_for_profile",
]
