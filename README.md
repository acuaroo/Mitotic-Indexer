# Mitotic Indexer
> *Made for a school biology project*

## Goal
Final goal:
***Given an image of cells from a WSI, identify the mitotic figures and return a mitotic index.***


Goal for this semester:
***Given an image of a cell, indetify if it is or isn't going through mitosis.***

## Process

`All main code is in /kaggle-main/`

`datagen.py`
- In `datagen.py` we use [*Marc Aubreville's implementation*](https://www.kaggle.com/code/marcaubreville/first-steps-with-the-mitos-wsi-ccmct-data-set) of fetching images & the annotations from the DICOM file & SQL file.
- Specify the `agreedClass`, `slide`, `limit`, and `size`, then fetch the cells that match that
- From this, we take each cell and take the needed data and push it into `train_labels`/`train_images`
- Data is shuffled and returned as `(train_images, train_labels), (test_images, test_labels)`

---

`model.py`
- `datagen.py` is now complete, `model.py` can now use the needed data

```python
from datagen import generate_final_data

(train_images, train_labels), (test_images, test_labels) = generate_final_data()
```

-  Add layers to `model.py`
```
Current List:
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- Flatten
- Dense
- Dense
```

- Setup `model.compile()`

```python
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"])
```

- Evaluating `model`

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
print(test_loss)
```

- Results!

```python
=================================================================
Total params: 129,570
Trainable params: 129,570
Non-trainable params: 0
_________________________________________________________________
188/188 [==============================] - 1s 4ms/step - loss: 0.1297 - accuracy: 0.9467
0.9466666579246521 <~ accuracy
0.12974701821804047 <~ loss
```
