from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Iterable


from PIL import Image
from git import Repo, Remote

from tmap_defectdetector.config.paths import DIR_DATASETS
from tmap_defectdetector.image_helpers import file_is_image
from tmap_defectdetector.logger import log

from tmap_defectdetector.datasets import ImageDataSet, AbstractDataSet


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

    DEFAULT_REPO_URL: Optional[str] = None
    DEFAULT_DATASET_ROOTDIR: Path = DIR_DATASETS

    def __init__(
        self,
        dataset_name: str,
        relative_sample_dir_paths: Iterable[os.PathLike],
        relative_label_file_paths: Iterable[os.PathLike],
        url: Optional[str] = DEFAULT_REPO_URL,
    ):
        """
        Class for cloning a (remote) git repository (e.g. containing a defect detection dataset).

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
        :param url: (optional) URL pointing to (remote) git repository to download.

        """
        self._dataset_name: str = dataset_name
        self._url: Optional[str] = url
        self.relative_label_file_paths = relative_label_file_paths
        self.relative_sample_dir_paths = relative_sample_dir_paths
        self._root_dataset_dir: Path = Path(self.DEFAULT_DATASET_ROOTDIR, dataset_name)

    @property
    def root_dataset_dir(self) -> Path:
        """Returns root directory for this dataset"""
        return self._root_dataset_dir

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def url(self) -> str:
        return self._url

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
        """Downloads defect detection dataset from (remote) repository URL."""
        if tgt_rootdir != self.root_dataset_dir:
            log.info(
                f"Changed root directory for git-downloaded dataset {type(self).__name__} "
                f"with name {self.dataset_name} "
                f"from {str(self.root_dataset_dir.resolve())} to {str(tgt_rootdir.resolve())}."
            )
            self._root_dataset_dir = tgt_rootdir

        repo, origin = self._initialize_repository(
            repo_dir=self.root_dataset_dir, remote_name=remote_name
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
    # Note that there is probably a far better more generalized approach than defining a class to
    # download/preprocess every dataset.
    # However, given that we probably don't have to use that many datasets and it would likely take to much
    # time to implement a generalizable way to do this right now, the choice was made to do it like this.

    DEFAULT_REPO_URL: Optional[str] = "https://github.com/jrbergen/elpv-dataset"
    DEFAULT_REPO_NAME: str = "dataset-elpv"

    def __init__(
        self,
        dataset_name: str = DEFAULT_REPO_NAME,
        relative_sample_dir_paths: Iterable[os.PathLike] = (Path("images"),),
        relative_label_file_paths: Iterable[os.PathLike] = (Path("labels.csv"),),
        url: Optional[str] = DEFAULT_REPO_URL,
    ):
        """
        (Down)loads datasets from https://github.com/jrbergen/Surface-Defect-Detection
        which contains example dataset
        datasets forked from https://github.com/Charmve/Surface-Defect-Detection.

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
        :param url: (optional str) url pointing to the remote repository for the ELPV dataset
        """
        super().__init__(dataset_name, relative_sample_dir_paths, relative_label_file_paths, url)

    def load(self) -> ImageDataSet:
        """
        Loads downloaded dataset into common ImageDataSet format.

        :returns: ImageDataSet
        """
        dataset = ImageDataSet()
        dataset = self._load_images(dataset)
        dataset = self._load_label_data(dataset)
        return dataset

    def _load_images(self, dataset: ImageDataSet) -> ImageDataSet:
        """
        Loads images into dataset for specified sample directories as
        dedfined in 'relative_sample_dir_paths' attribute.

        :param dataset: ImageDataset instance.
        """
        image_dirs = [Path(self.root_dataset_dir, p) for p in self.relative_sample_dir_paths]

        for imgdir in image_dirs:
            for file in imgdir.rglob("*.*"):
                if file.is_file() and file_is_image(file):
                    with Image.open(file) as imgobj:
                        dataset += imgobj

        log.info(
            f"Loaded dataset {self.dataset_name!r} ({type(self).__name__}) with {len(dataset)} images."
        )
        return dataset

    def _load_label_data(self, dataset: ImageDataSet) -> ImageDataSet:
        """
        Loads images into dataset for specified sample directories as
        defined in 'relative_sample_dir_paths' attribute.

        :param dataset: ImageDataset instance.
        """
        label_files = [Path(self.root_dataset_dir, lf) for lf in self.relative_label_file_paths]

        for labelpath in label_files:
            if labelpath.is_file():
                dataset.add_label_data(labelpath)

        log.info(
            f"Loaded label data for dataset {self.dataset_name!r} "
            f"({type(self).__name__} from {len(label_files)} label files."
        )

        return dataset
