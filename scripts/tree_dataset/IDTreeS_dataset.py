import glob
import time

import geopandas
import numpy as np
import pandas as pd

from tree_dataset import TreeDataset


class IDTreeSDataset(TreeDataset):
    """
    Extract tree crowns from the dataset from the IDTreeS competition

    Args:
        rgb_paths: paths to the RGB images (GeoTIFF files in the
            `train/RemoteSensing/RGB` folder).
        bboxes_paths: paths to the individual tree crowns bounding
            boxes (shapefiles in the `train/ITC` folder).
        classes_path: path to the file with the tree classification
            (`train/Field/train_data.csv` file).
    """
    def __init__(
            self,
            rgb_paths,
            bboxes_paths,
            classes_path,
            **kwargs
    ):

        classes = _load_classes(classes_path)
        bboxes = _load_bboxes(bboxes_paths, classes)

        super().__init__(
            rgb_paths,
            bboxes,
            classes,
            **kwargs
        )


def _load_classes(classes_path):
    """ Load data classes. """
    classes = pd.read_csv(classes_path)

    # drop duplicate entries - at least repeated entries
    # seem to have the same class label
    classes = classes.drop_duplicates(subset='indvdID')

    classes = classes.set_index('indvdID')
    return classes['taxonID']


def _load_bboxes(bboxes_paths, classes):
    """
    Load bounding boxes, retaining the only tree IDs for which
    a class is available (classes need to be loaded first).
    Multiple boxes can exist for a given tree ID (presumably
    trees crossing the edges between images).
    """
    bboxes = pd.concat([
        geopandas.read_file(p)
        for p in bboxes_paths
    ])

    bboxes = bboxes.set_index('indvdID')

    # use GeoDataFrame as left dataframe (preserves object type)
    bboxes = pd.merge(
        bboxes,
        classes,
        how='right',
        on='indvdID'
    )
    return bboxes['geometry']


if __name__ == "__main__":
    # Dataset from https://zenodo.org/record/3934932, download at:
    # https://zenodo.org/record/3934932/files/IDTREES_competition_test_v2.zip
    # Unzipping should cread folder 'train'
    rgb_paths = glob.glob('./train/RemoteSensing/RGB/*.tif')
    bboxes_paths = glob.glob('./train/ITC/train_*.shp')
    classes_path = './train/Field/train_data.csv'

    print("Loading all cutouts ...")
    ds = IDTreeSDataset(
        rgb_paths,
        bboxes_paths,
        classes_path,
        min_pixel_size=32,
        max_pixel_size=100,
        pixel_size=100,
        min_sample_size=10,
        augment_data=False
    )

    s = time.time()
    ids, labels, cutouts = ds.get_cutouts()
    print(f"Executed in {time.time() - s}")

    # TODO: the following could be implemented as tests
    print(f"Num. of samples: {len(ids)}")  # 515
    ll, cc = np.unique(labels, return_counts=True)
    print(f"Num. classes: {len(ll)}")  # 9
    print(f"Min. population, Max. population: {min(cc), max(cc)}")  # (14, 115)
    print(f"Image shape: {cutouts.shape[1:]}")  # (100, 100, 3)
