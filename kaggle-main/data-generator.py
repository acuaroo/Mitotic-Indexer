import os
import io
import sqlite3 as sql
import numpy as np
import pydicom as pyd
from array import *
from tempfile import TemporaryFile
from pydicom.encaps import decode_data_sequence
from PIL import Image, ImageOps
from PIL import ImageDraw


# for dirname, _, filenames, in os.walk('dicom-set'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#

sq_db = sql.connect('dicom-set/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite')
cursor = sq_db.cursor()

pixel_matrix_columns = [0x48, 0x6]
pixel_matrix_rows = [0x48, 0x7]
round_up = 0.5

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

#------------WIP ZONE------------#
#slide 4 = c3eb4b8382b470dd63a9.dcm
final_ds = DicomDataset('dicom-set/c3eb4b8382b470dd63a9.dcm')

slide = 4
agrClass = 4
limit = 100
outfile = TemporaryFile()

cells = cursor.execute(f"""SELECT Annotations.uid, Annotations.slide, Annotations.agreedClass, Annotations_coordinates.coordinateX, Annotations_coordinates.coordinateY, Annotations_coordinates.uid FROM 'Annotations_coordinates' JOIN 'Annotations' ON Annotations.uid = Annotations_coordinates.uid where agreedClass=={agrClass} and Annotations.slide=={slide} LIMIT 0,{limit}""").fetchall()

train_images = np.zeros((100,40,40),dtype=float)
train_labels = np.zeros(100,dtype=int)

#print(train_images.shape)
#looping through the cells
iteration = 0
filename = "class_4_training.npz"

for cell in cells:
    #cell[3] is xcoord cell[4] is ycoord cell[2] is
    location = (cell[3]-20, cell[4]-20)
    size = (40, 40)
    img = Image.fromarray(final_ds.read_region(location, size))
    grayscale = ImageOps.grayscale(img)
    np_im = np.array(grayscale)
    train_images[iteration,:,:] = np_im
    train_labels[iteration] = cell[2]
    #print(iteration)
    iteration += 1

np.savez(filename, train_images, train_labels)
print("data saved in "+filename)

# draw = ImageDraw.Draw(img)
# r=25

# draw.ellipse([(location[0]-r, location[1]-r),(location[0]+r,location[1]+r)],outline=(0,0,255,255))
#
# img.show()