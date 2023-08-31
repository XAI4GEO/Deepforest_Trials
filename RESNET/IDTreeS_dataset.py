import glob
import time

import geopandas
import pandas as pd
import rioxarray
import numpy as np

from rioxarray.exceptions import NoDataInBounds
from sklearn.preprocessing import LabelEncoder


MIN_PIXEL_SIZE_DEF = 32
MAX_PIXEL_SIZE_DEF = 100
PIXEL_SIZE_DEF = 100
MIN_SAMPLE_SIZE_DEF = 10


class IDTreeSDataset:
    def __init__(
            self,
            rgb_paths,
            bboxes_paths,
            classes_path,
            min_pixel_size=None,
            max_pixel_size=None,
            pixel_size=None,
            min_sample_size=None,
        ):
        self.rgb_paths = rgb_paths
        self.bboxes_paths = bboxes_paths
        self.classes_path = classes_path

        self.min_pixel_size = min_pixel_size if min_pixel_size is not None \
            else MIN_PIXEL_SIZE_DEF
        self.max_pixel_size = max_pixel_size if max_pixel_size is not None \
            else MAX_PIXEL_SIZE_DEF
        self.pixel_size = pixel_size if pixel_size is not None \
            else PIXEL_SIZE_DEF
        self.min_sample_size = min_sample_size if min_sample_size is not None \
            else MIN_SAMPLE_SIZE_DEF

        self.classes = None
        self.bboxes = None

    def _load_classes(self):
        """ Load data classes. """
        classes = pd.read_csv(self.classes_path)

        # drop duplicate entries - at least repeated entries
        # seem to have the same class label
        classes = classes.drop_duplicates(subset='indvdID')

        classes = classes.set_index('indvdID')
        self.classes = classes['taxonID']

    def _load_bboxes(self):
        """
        Load bounding boxes, retaining the only tree IDs for which
        a class is available (classes need to be loaded first).
        Multiple boxes can exist for a given tree ID (presumably
        trees crossing the edges between images).
        """
        bboxes = pd.concat([
            geopandas.read_file(p)
            for p in self.bboxes_paths
        ])

        bboxes = bboxes.set_index('indvdID')

        # use GeoDataFrame as left dataframe (preserves object type)
        bboxes = pd.merge(
            bboxes,
            self.classes,
            how='right',
            on='indvdID'
        )
        self.bboxes = bboxes['geometry']

    def _generate_cutouts_for_image(self, rgb_path):
        """ Extract all cutouts for a given image. """

        if self.classes is None:
            self._load_classes()
        if self.bboxes is None:
            self._load_bboxes()

        rgb = rioxarray.open_rasterio(rgb_path, masked=True)

        assert self.bboxes.crs == rgb.rio.crs

        # select relevant bboxes
        xmin, ymin, xmax, ymax = rgb.rio.bounds()
        bboxes_clip = self.bboxes.cx[xmin:xmax, ymin:ymax]
        bboxes_clip = bboxes_clip[~bboxes_clip.is_empty]

        for indvdID, bbox in bboxes_clip.items():

            # some polygons have very small intersections
            try:
                cutout = rgb.rio.clip([bbox], drop=True)
            except NoDataInBounds:
                continue

            # drop large or small cutouts
            if (
                _large_image(cutout.data, self.max_pixel_size)
                or _small_image(cutout.data, self.min_pixel_size)
            ):
                continue

            label = self.classes[indvdID]
            img = _pad_image(cutout.data, self.pixel_size)

            yield indvdID, label, img

    def get_cutouts(self):
        """ Load cutouts from all images. """
        tree_ids = []
        labels = []
        imgs = []
        for rgb_path in self.rgb_paths:
            cutouts = self._generate_cutouts_for_image(rgb_path)
            for tree_id, label, img in cutouts:
                tree_ids.append(tree_id)
                labels.append(label)
                imgs.append(img)

        tree_ids, labels, imgs = _balance_classes(
            np.asarray(tree_ids),
            np.asarray(labels),
            np.stack(imgs),
            self.min_sample_size
        )

        # convert class labels to categorical values
        le = LabelEncoder()
        le.fit(labels)
        labels_categorical = le.transform(labels)
        return tree_ids, labels_categorical, imgs


def _balance_classes(tree_ids, labels, imgs, min_sample_size):
    """
    Balance class compositions by dropping classes with
    few samples and by performing data augmentation.
    """
    labels_unique, counts = np.unique(labels, return_counts=True)
    labels_to_keep = labels_unique[counts >= min_sample_size]
    mask = np.isin(labels, labels_to_keep)

    # data augmentation here?
    return (
        tree_ids[mask],
        labels[mask],
        imgs[mask],
    )


def _large_image(data, max_pixel_size):
    if max(data.shape[1:]) > max_pixel_size:
        return True
    return False


def _small_image(data, min_pixel_size):
    if min(data.shape[1:]) <= min_pixel_size:
        return True
    return False


def _pad_image(data, pixel_size):
    """ Pad each image """
    pad_width_x1 = np.floor((pixel_size - data.shape[1])/2).astype(int)
    pad_width_x2 = pixel_size - data.shape[1] - pad_width_x1
    pad_width_y1 = np.floor((pixel_size - data.shape[2])/2).astype(int)
    pad_width_y2 = pixel_size - data.shape[2] - pad_width_y1
    data = np.pad(data, pad_width=[(0, 0),(pad_width_x1, pad_width_x2),(pad_width_y1, pad_width_y2)], mode='constant')
    return data


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
