"""
Contains abstract baseclasses for downloading datasets,
as well as implementations thereof for downloading specific datasets.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Callable

from git import Repo, Remote
from tqdm import tqdm

from tmap_defectdetector.config.paths import DIR_DATASETS
from tmap_defectdetector.logger import log

from tmap_defectdetector.datasets import AbstractDataSet


class AbstractDatasetDownloader(ABC):
    """
    Defect data could be downloaded in various ways (e.g. from disk, from web).
    This abstract baseclass defines the methods that a DatasetDownloader should have
    (at least a method for downloading the dataset, and one to load it into a common (internal) format).
    """

    @abstractmethod
    def download(self) -> None:
        """This method should handle the download portion for the dataset."""
        pass

    @abstractmethod
    def load(self) -> AbstractDataSet:
        """This method should handle the loading of the dataset into a common format."""
        pass


class DatasetDownloaderGit(AbstractDatasetDownloader):

    DEFAULT_DATASET_ROOTDIR: Path = DIR_DATASETS

    def __init__(
        self,
        repo_url: str,
        dataset_name: str,
        relative_sample_dir_paths: Iterable[os.PathLike],
        relative_label_file_paths: Iterable[os.PathLike],
    ):
        """
        Class for cloning a (remote) git repository (e.g. containing a defect detection dataset).

        :param repo_url: URL pointing to (remote) git repository to download.
        :param dataset_name: name for this dataset; e.g. the repository's name.
        :param relative_sample_dir_paths: iterable containing one or more paths
            to director[y|ies] containing the training data relative to the root repository directory.
            e.g. if the repository is called "dataset_repo" and the directories containing training images
            are in "dataset_repo/training_data/image_set1" and "dataset_repo/training_data/image_set2",
            then you could pass ["dataset_repo/training_data/image_set1", "dataset_repo/training_data/image_set2"]
            to this argument.
        :param relative_label_file_paths: iterable containing one or more paths
            to a file containing the labels (e.g. a CSV file) relative to the root repository directory.
            e.g. if the repository is called "dataset_repo" and the 'labels' file is in
            "dataset_repo/labels/labels_set1.csv", then you should pass ["labels/image_set1.csv"]
            to this argument.
        """
        self._repo_url: str = repo_url
        self._dataset_name: str = dataset_name
        self.relative_label_file_paths = relative_label_file_paths
        self.relative_sample_dir_paths = relative_sample_dir_paths
        self._dataset_dir: Path = Path(self.DEFAULT_DATASET_ROOTDIR, dataset_name)

    @property
    def dataset_dir(self) -> Path:
        """Returns root directory w.r.t. this dataset"""
        return self._dataset_dir

    @property
    def dataset_name(self) -> str:
        """Returns name of this dataset. Is used for dataset's root directory name and identification."""
        return self._dataset_name

    @property
    def repo_url(self) -> str:
        """URL pointing to repository, cannot be changed; instantiate a new object instead."""
        return self._repo_url

    def load(self) -> AbstractDataSet:
        raise NotImplementedError(
            "Conversion of this dataset to 'datasets.DataSet' format not yet implemented."
        )

    def download(
        self,
        tgt_rootdir: Path = DIR_DATASETS,
        remote_name: str = "origin",
        target_branch: str = "master",
    ) -> None:
        """
        Downloads defect detection dataset from (remote) repository URL (URL is specified in constructor).

        :param tgt_rootdir: Parent of directory which will contain the repository (and its) dataset.
        :param remote_name: (optional) name of remote repository (Defaults to "origin")
        :param target_branch: (optioanl) name of target branch (Defaults to "master")
        """

        # Change target root directory if it differs from the current.
        if tgt_rootdir != self._dataset_dir.parent:
            new_dataset_dir = Path(tgt_rootdir, self.dataset_name)

            log.info(
                f"Changed directory for git-downloaded dataset {type(self).__name__} "
                f"with name {self.dataset_name} "
                f"from {str(self.dataset_dir.resolve())} to {str(new_dataset_dir.resolve())}."
            )
            self._dataset_dir = new_dataset_dir

        # Initialize, fetch repository and checkout specified target branch.
        repo, origin = self._initialize_repository(
            repo_dir=self.dataset_dir, remote_name=remote_name
        )

        self._fetch(origin=origin)
        self._checkout_remote_branch(repository=repo, origin=origin)

    def _initialize_repository(
        self, repo_dir: str | os.PathLike, remote_name: str
    ) -> tuple[Repo, Remote]:
        """Used internally. Initializes repository in the specified directory for a given remote name."""
        repo = Repo.init(repo_dir)
        try:
            origin = repo.remotes[remote_name]
        except (KeyError, IndexError):
            # Repo doesn't have this remote yet... Initialize it.
            origin = repo.create_remote(remote_name, self.repo_url)
        return repo, origin

    def _fetch(self, origin: Remote) -> Remote:
        """Used internally. Fetches repository from a Remote"""
        log.info(f"Fetching dataset from remote git (url={self.repo_url!r})...")
        origin.fetch()
        log.info("Dataset fetch from remote git complete!")
        return origin

    @classmethod
    def _checkout_remote_branch(
        cls, repository: Repo, origin: Remote, main_master_branch: str = "master"
    ) -> Repo:
        """Used internally. Checks out repository for specified branch and remote."""
        # Setup a local tracking branch of a remote branch
        repository.create_head(
            main_master_branch, origin.refs.master
        )  # create local branch from remote "master"
        repository.heads.master.set_tracking_branch(
            origin.refs.master
        )  # set local "master" to track remote "master
        repository.heads.master.checkout()  # checkout local "master" to working tree
        return repository

    @property
    def label_paths(self) -> list[Path]:
        """
        Returns list of absolute filepaths which should point to this dataset's
        file(s) which contain the dataset labels.
        """
        return [
            Path(self.dataset_dir, relative_path)
            for relative_path in self.relative_label_file_paths
        ]

    @property
    def data_sample_dirs(self) -> list[Path]:
        """
        Returns list of absolute filepaths which should point to this dataset's
        directories containing sample files for training (e.g. sample images).
        """
        return [
            Path(self.dataset_dir, relative_path)
            for relative_path in self.relative_sample_dir_paths
        ]

    def get_data_files(
        self,
        filechecker_functions: Callable[[Path], bool] = Path.is_file,
    ) -> list[Path]:
        """
        Looks in sample director[y|ies] for this dataset, gathering all files which
        comply with the file checking function passed to the 'filechecker_function' parameter.
        """
        log.info("Updating dataset's training sample paths...")
        data_paths = []
        # Currently this crudely finds all image files in any of the (sub)directories in the image_dirs variable.
        for sample_dir in self.data_sample_dirs:
            for file in tqdm(sample_dir.rglob("*.*"), desc="Finding images...", unit="images"):
                if filechecker_functions(file):
                    data_paths.append(file)
        return data_paths


class DatasetDownloaderELPV(DatasetDownloaderGit):
    # Note that there is probably a far better more generalized approach than defining a class to
    # download/preprocess every dataset.
    # However, given that we probably don't have to use that many datasets and it would likely take to much
    # time to implement a generalizable way to do this right now, the choice was made to do it like this.

    DEFAULT_REPO_NAME: str = "dataset-elpv"

    def __init__(
        self,
        dataset_name: str = DEFAULT_REPO_NAME,
        relative_sample_dir_paths: Iterable[os.PathLike] = (Path("images"),),
        relative_label_file_paths: Iterable[os.PathLike] = (Path("labels.csv"),),
        repo_url: str = "https://github.com/jrbergen/elpv-dataset",
    ):
        """
        Downloads datasets from https://github.com/jrbergen/elpv-dataset
        which contains example datasets forked from https://github.com/zae-bayern/elpv-dataset

        :param repo_url: (optional) url pointing to the remote repository for the ELPV dataset
            (Defaults to "https://github.com/jrbergen/elpv-dataset").
        :param dataset_name: (optional str) name for this dataset
            (defaults to DatasetDownloaderELPV.DEFAULT_REPO_NAME).
        :param relative_sample_dir_paths: (optional) iterable containing one or more paths
            to director[y|ies] containing the training data relative to the root repository directory.
            e.g. if the repository is called "dataset_repo" and the directories containing training images
            are in "dataset_repo/training_data/image_set1" and "dataset_repo/training_data/image_set2",
            then you could pass ["dataset_repo/training_data/image_set1", "dataset_repo/training_data/image_set2"]
            to this argument. Defaults to "images" (==dir w/ training data the ELPV dataset).
        :param relative_label_file_paths: (optional) iterable containing one or more paths
            to a file containing the labels (e.g. a CSV file) relative to the root repository directory.
            e.g. if the repository is called "dataset_repo" and the 'labels' file is in
            "dataset_repo/labels/labels_set1.csv", then you should pass ["labels/image_set1.csv"]
            to this argument. Defaults to "labels.csv" (==file w/ labels for the ELPV dataset).
            Warning: will assume every file in this relative directory is a file containing labels.

        """
        super().__init__(
            repo_url=repo_url,
            dataset_name=dataset_name,
            relative_sample_dir_paths=relative_sample_dir_paths,
            relative_label_file_paths=relative_label_file_paths,
        )

        # The image loading logic which was here before is now part of the ImageDataSetELPV class.
        # We may want to alter this class later (e.g. remove it entirely and find a better way to
        # make its DataSetDownloader parent class suited to donwloading the ELPV dataset specifically).
