import glob
import time

import geopandas
import numpy as np
import pandas as pd

from tree_dataset import TreeDataset


class IDTreeSDataset(TreeDataset):
    def __init__(
            self,
            rgb_paths,
            bboxes_paths,
            classes_path,
            min_pixel_size=None,
            max_pixel_size=None,
            pixel_size=None,
            min_sample_size=None,
            augment_data=False
    ):

        classes = _load_classes(classes_path)
        bboxes = _load_bboxes(bboxes_paths, classes)

        super().__init__(
            rgb_paths,
            bboxes,
            classes,
            min_pixel_size=min_pixel_size,
            max_pixel_size=max_pixel_size,
            pixel_size=pixel_size,
            min_sample_size=min_sample_size,
            augment_data=augment_data,
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
    rgb_paths = glob.glob('./train/RemoteSensing/RGB/*.tif')
    bboxes_paths = glob.glob('./train/ITC/train_*.shp')
    classes_path = './train/Field/train_data.csv'

    print("Loading all cutouts ...")
    ds = IDTreeSDataset(rgb_paths, bboxes_paths, classes_path)

    s = time.time()
    ids, labels, cutouts = ds.get_cutouts()
    print(f"Executed in {time.time() - s}")

    print(f"Num. of samples: {len(ids)}")
    ll, cc = np.unique(labels, return_counts=True)
    print(f"Num. classes: {len(ll)}")
    print(f"Min. population, Max. population: {min(cc), max(cc)}")
    print(f"Image shape: {cutouts.shape[1:]}")
