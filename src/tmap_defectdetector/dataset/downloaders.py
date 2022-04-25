"""Downloader class implementations for downloading specific datasets are defined here."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from src.tmap_defectdetector.dataset.base.downloaders_base import DataSetDownloaderGit


class DataSetDownloaderELPV(DataSetDownloaderGit):
    # Note that there is probably a far better more generalized approach than defining a class to
    # download/preprocess every dataset.
    # However, given that we probably don't have to use that many datasets and it would likely take to much
    # time to implement a generalizable way to do this right now, the choice was made to do it like this.

    # Default repository name for this DataSetDownloader subclass
    DEFAULT_REPO_NAME: str = "dataset-elpv"

    # Description of the type of sample data used for e.g. progress bars,
    # should not impact functionality
    DEFAULT_DATASET_UNITS: str = "image samples"

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
            (defaults to DataSetDownloaderELPV.DEFAULT_REPO_NAME).
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
