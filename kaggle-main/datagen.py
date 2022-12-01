import os
import io
import random

import sqlite3 as sql
import numpy as np
import pydicom as pyd

from pathlib import Path
from array import *
from pydicom.encaps import decode_data_sequence
from PIL import Image, ImageOps
from PIL import ImageDraw

#getting the SQL annotations
sq_db = sql.connect('dicom-set/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite')
cursor = sq_db.cursor()

#some of the data from the DICOM set
pixel_matrix_columns = [0x48, 0x6]
pixel_matrix_rows = [0x48, 0x7]
round_up = 0.5

def generate_final_data():
    #-------------------data prep-------------------#

    # below is Marc Aubreville's implementation of fetching images & the annotations
    # https://www.kaggle.com/code/marcaubreville/first-steps-with-the-mitos-wsi-ccmct-data-set
    class DicomDataset:
        def __init__(self, file):
            self._ds = pyd.read_file(file)
            self.image_size = (self._ds[pixel_matrix_columns].value, self._ds[pixel_matrix_rows].value)
            self.tile_size = (self._ds.Columns, self._ds.Rows)
            self.columns = round(round_up + (self.image_size[0] / self.tile_size[0]))
            self.rows = round(round_up + (self.image_size[1] / self.tile_size[1] ))
            self._dsequence = decode_data_sequence(self._ds.PixelData)

        def get_tile(self, position):
            ds_seq = self._dsequence[position]
            return np.array(Image.open(io.BytesIO(ds_seq)))

        def get_id(self, pixel_x:int, pixel_y:int) -> (int, int, int):

            id_x = round(-0.5+(pixel_x/self.tile_size[1]))
            id_y = round(-0.5+(pixel_y/self.tile_size[0]))

            return (id_x,id_y), pixel_x-(id_x*self.tile_size[0]), pixel_y-(id_y*self.tile_size[1])

        def imagepos_to_id(self, image_pos:tuple):
            id_xn, id_yn = image_pos
            return id_xn+(id_yn * self.columns)

        @property
        def dimensions(self):
            return self.image_size

        def read_region(self, locus:tuple, dim:tuple):
            lu, lu_xo, lu_yo = self.get_id(*list(locus))
            rl, rl_xo, rl_yo = self.get_id(*[sum(x) for x in zip(locus, dim)])
            # generate big image
            bigimg = np.zeros(((rl[1]-lu[1]+1)*self.tile_size[0], (rl[0]-lu[0]+1)*self.tile_size[1], self._ds[0x0028, 0x0002].value), np.uint8)
            for xi, xgridc in enumerate(range(lu[0],rl[0]+1)):
                for yi, ygridc in enumerate(range(lu[1],rl[1]+1)):
                    if (xgridc<0) or (ygridc<0):
                        continue
                    bigimg[yi*self.tile_size[0]:(yi+1)*self.tile_size[0],
                    xi*self.tile_size[1]:(xi+1)*self.tile_size[1]] = \
                        self.get_tile(self.imagepos_to_id((xgridc,ygridc)))

            # crop big image
            return bigimg[lu_yo:lu_yo+dim[1],lu_xo:lu_xo+dim[0]]

    def concat_images(img_paths, sz, shape=None):
        width, height = sz
        images = map(Image.open, img_paths)
        images = [ImageOps.fit(im, sz, Image.Resampling.LANCZOS)
                  for im in images]

        shape = shape if shape else (1, len(images))
        image_size = (width * shape[1], height * shape[0])
        new_image = Image.new('RGB', image_size)

        for row in range(shape[0]):
            for col in range(shape[1]):
                offset = width * col, height * row
                idx = row * shape[1] + col
                new_image.paste(images[idx], offset)

        return new_image

    #-------------------data generation-------------------#

    #slide 4 = c3eb4b8382b470dd63a9.dcm
    final_ds = DicomDataset('dicom-set/c3eb4b8382b470dd63a9.dcm')
    def generate_data(slide, agr_class, lim, size_set):
        #sql command takes all of the cells with the restrictions above
        #and puts the data together (so combines cell location & cell class)
        cells = cursor.execute(f"""SELECT Annotations.uid, Annotations.slide, Annotations.agreedClass, Annotations_coordinates.coordinateX, Annotations_coordinates.coordinateY, Annotations_coordinates.uid FROM 'Annotations_coordinates' JOIN 'Annotations' ON Annotations.uid = Annotations_coordinates.uid where agreedClass=={agr_class} and Annotations.slide=={slide} LIMIT 0,{limit}""").fetchall()

        tr_images = np.zeros((lim,size_set,size_set),dtype=float)
        tr_labels = np.zeros(lim,dtype=int)

        iteration = 0

        for cell in cells:
            #cell[3] is xcoord
            #cell[4] is ycoord
            #cell[2] is cells agreedClass

            
            # binary - div 2, -2, *-1 (or abs)
            location = (cell[3]-20, cell[4]-20)
            size = (size_set, size_set)
            img = Image.fromarray(final_ds.read_region(location, size))

            grayscale = ImageOps.grayscale(img)
            np_im = np.array(grayscale)
            tr_images[iteration,:,:] = np_im
            tr_labels[iteration] = cell[2]

            iteration += 1

        # if you want to see what the data looks like
        # uncomment this code to get a 9x9 grid as an image

        # image_paths = [os.path.join(folder, f)
        #                for f in os.listdir(folder) if f.endswith(".jpg")]
        #
        # image = concat_images(image_paths, (size_set*10, size_set*10), (9, 9))
        # image.save(str(agr_class)+"_class.jpg")

        return tr_labels, tr_images

    limit = 10000
    train_percent = 0.7

    c2_train_labels, c2_train_images = generate_data(4, 2, limit, 40)
    #same as above, just doing class 4 (other cells)
    c4_train_labels, c4_train_images = generate_data(4, 4, limit, 40)

    #turns 4 -> 0 and 2 -> 1 to make it binary
    c4_train_labels = c4_train_labels*0
    c2_train_labels = c2_train_labels*(1/2)

    #shuffling data
    all_images = np.concatenate((c2_train_images, c4_train_images), axis=0)
    all_labels = np.concatenate((c2_train_labels, c4_train_labels), axis=0)

    shuffler = np.random.permutation(len(all_images))
    all_images = all_images[shuffler]
    all_labels = all_labels[shuffler]

    #rounding the limit*2 (all data length) and spliting it up into training vs testing
    rounded_minmax = round((limit*2) * train_percent)

    train_images = all_images[:rounded_minmax,:,:]
    train_labels = all_labels[:rounded_minmax]

    test_images = all_images[rounded_minmax:,:,:]
    test_labels = all_labels[rounded_minmax:]

    print(train_images.shape)
    print(test_images.shape)

    #final return value
    #data is now fully prepared for our modle.py file
    return (train_images, train_labels), (test_images, test_labels)
