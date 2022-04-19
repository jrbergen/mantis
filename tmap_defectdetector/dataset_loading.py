from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from git import Repo, Remote

from tmap_defectdetector.config.paths import DIR_DATASETS
from tmap_defectdetector.logger import log

if TYPE_CHECKING:
    from tmap_defectdetector.datasets import DataSet


class DatasetDownloader(ABC):
    """
    Defect data could be downloaded in various ways (e.g. from disk, from web).
    This prototype (the 'ducktyping' alternative to an abstract base class) defines the methods
    that a DataLoader should have.
    """

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def load(self) -> DataSet:
        pass


class DatasetDownloaderGit(DatasetDownloader):
    """Baseclass for downloading defect detection dataset from (remote) git repository."""

    DEFAULT_REPO_URL: Optional[str] = None
    DEFAULT_DATASET_ROOTDIR: Optional[Path] = DIR_DATASETS

    def __init__(
        self,
        dataset_name: str,
        url: Optional[str] = DEFAULT_REPO_URL,
    ):
        self._dataset_name: str = dataset_name
        self._url: Optional[str] = url

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def url(self) -> str:
        return self._url

    def load(self) -> DataSet:
        raise NotImplementedError(
            "Conversion of this dataset to 'datasets.DataSet' format not yet implemented."
        )

    def download(
        self,
        tgt_rootdir: Path = DIR_DATASETS,
        remote_name: str = "origin",
        target_branch: str = "master",
    ) -> None:
        """Downloads defect detection dataset from (remote) repository URL."""
        repo, origin = self._initialize_repository(
            repo_dir=tgt_rootdir / self._dataset_name, remote_name=remote_name
        )
        self._fetch(origin=origin)
        self._checkout_remote_branch(repository=repo, origin=origin)

    def _initialize_repository(
        self, repo_dir: str | os.PathLike, remote_name: str
    ) -> tuple[Repo, Remote]:
        repo = Repo.init(repo_dir)
        try:
            origin = repo.remotes[remote_name]
        except (KeyError, IndexError):
            # Repo doesn't have this remote yet... Initialize it.
            origin = repo.create_remote(remote_name, self.url)
        return repo, origin

    def _fetch(self, origin: Remote) -> Remote:
        log.info(f"Fetching dataset from remote git (url={self.url!r})...")
        origin.fetch()
        log.info("Dataset fetch from remote git complete!")
        return origin

    @classmethod
    def _checkout_remote_branch(
        cls, repository: Repo, origin: Remote, main_master_branch: str = "master"
    ) -> Repo:
        # Setup a local tracking branch of a remote branch
        repository.create_head(
            main_master_branch, origin.refs.master
        )  # create local branch from remote "master"
        repository.heads.master.set_tracking_branch(
            origin.refs.master
        )  # set local "master" to track remote "master
        repository.heads.master.checkout()  # checkout local "master" to working tree
        return repository


class DatasetDownloaderELPV(DatasetDownloaderGit):
    """
    (Down)loads datasets from https://github.com/jrbergen/Surface-Defect-Detection
    which contains example dataset
    datasets forked from https://github.com/Charmve/Surface-Defect-Detection.
    """

    DEFAULT_REPO_URL: Optional[str] = "https://github.com/jrbergen/elpv-dataset"
    DEFAULT_REPO_NAME: str = "dataset-elpv"

    def __init__(
        self, dataset_name: str = DEFAULT_REPO_NAME, url: Optional[str] = DEFAULT_REPO_URL
    ):
        super().__init__(dataset_name, url)
