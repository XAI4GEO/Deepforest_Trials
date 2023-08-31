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


class IDTreeSDataset:
    
    def __init__(
            self, 
            rgb_paths, 
            bboxes_paths, 
            classes_path,
            min_pixel_size=None,
            max_pixel_size=None,
            pixel_size=None,
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

        self.classes = None
        self.bboxes = None

    def _load_classes(self):
        """ Load classes and convert them to categorical values. """
        classes = pd.read_csv(self.classes_path)
        
        # drop duplicate entries - repeated labels appear to be the same
        classes = classes.drop_duplicates(subset='indvdID')

        # convert classes to categorical values
        le = LabelEncoder()
        le.fit(classes.taxonID)
        classes['categorical_label'] = le.transform(classes.taxonID)

        classes = classes.set_index('indvdID')
        self.classes = classes['categorical_label']

    def _load_bboxes(self):
        """ 
        Load bounding boxes, retaining the only tree IDs for which
        a class is available (classes need to be loaded first). 
        Multiple boxes can exist for a given tree ID (presumably
        trees at the boundary between images). 
        """
        bboxes = pd.concat([
            geopandas.read_file(p)
            for p in self.bboxes_paths
        ])

        bboxes = bboxes.set_index('indvdID')
        # if GeoDataFrame is left, the data structure is maintained
        bboxes = pd.merge(bboxes, self.classes, how='right', on='indvdID')
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

            # some polygons have very small intersections with the images
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

    def generate_cutouts(self):
        """ 
        Return cutouts extracted from all images (as a generator).
        """
        for rgb_path in self.rgb_paths:
            cutouts = self._generate_cutouts_for_image(rgb_path)
            for tree_id, label, img in cutouts:
                yield tree_id, label, img
    
    def get_all_cutouts(self):
        """ 
        Load all cutouts from all images and return them as a 
        Numpy array. 
        """
        tree_ids = []
        labels = []
        imgs = []
        for rgb_path in self.rgb_paths:
            cutouts = self._generate_cutouts_for_image(rgb_path)
            for tree_id, label, img in cutouts:
                tree_ids.append(tree_id)
                labels.append(label)
                imgs.append(img)
        return (
            np.asarray(tree_ids), 
            np.asarray(labels),
            np.stack(imgs),
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

    # use as a generator ...
    print("Run as a generator ...")
    ds = IDTreeSDataset(rgb_paths, bboxes_paths, classes_path)
    cutouts = ds.generate_cutouts()

    s = time.time()
    for cutout in ds.generate_cutouts():
        _, _, _ = cutout
    print(f"Executed in {time.time() - s}")

    # ... or load all data at once
    print("Load all cutouts ...")
    ds = IDTreeSDataset(rgb_paths, bboxes_paths, classes_path)

    s = time.time()
    cutouts = ds.get_all_cutouts()
    print(f"Executed in {time.time() - s}")
    print(cutouts[-1].shape)
