# Mitotic Indexer
> *Made for a school biology project*
---

## Goal
Final goal
> Given an image of cells from a WSI, identify the mitotic figures and return a mitotic index.

Goal for this semester
> Given an image of a cell, indetify if it is or isn't going through mitosis

---

## Process
---
`All main code is in /kaggle-main/`
---

- In `datagen.py` we use [*Marc Aubreville's implementation*](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set) of fetching images & the annotations from the DICOM file & SQL file.
- Specify the `agreedClass`, `slide`, `limit`, and `size`, then fetch the cells that match that
- From this, we take each cell and take the needed data and push it into `train_labels`/`train_images`
- Data is shuffled and returned as `(train_images, train_labels), (test_images, test_labels)`

- `datagen.py` is now complete, `model.py` can now use the needed data

