DSTL
====

https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/

Object types
------------

- 0: Buildings - large building, residential, non-residential, fuel storage facility, fortified building
- 1: Misc. Manmade structures
- 2: Road
- 3: Track - poor/dirt/cart track, footpath/trail
- 4: Trees - woodland, hedgerows, groups of trees, standalone trees
- 5: Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
- 6: Waterway
- 7: Standing water
- 8: Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
- 9: Vehicle Small - small vehicle (car, van), motorbike


Making submission
-----------------

Train a CNN (choose number of epochs and other hyper-params running without
``--all``)::

    $ ./train.py debug12-all-ep4 --all --hps n_epochs=4

Make submission file (be sure to pass hyperparameters that influence CNN structure)::

    $ ./make_submission.py debug12-all-ep4 sub_th0.4_debug12-all-ep4 --threshold 0.4

