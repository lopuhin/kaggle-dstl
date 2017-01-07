DSTL
====

https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/

Object types
------------

- 1: Buildings - large building, residential, non-residential, fuel storage facility, fortified building
- 2: Misc. Manmade structures
- 3: Road
- 4: Track - poor/dirt/cart track, footpath/trail
- 5: Trees - woodland, hedgerows, groups of trees, standalone trees
- 6: Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
- 7: Waterway
- 8: Standing water
- 9: Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
- 10: Vehicle Small - small vehicle (car, van), motorbike


Making submission
-----------------

Train a CNN (choose number of epochs and other hyper-params running without
``--all``)::

    $ ./train.py debug12-all-ep4 --all --hps n_epochs=4

Make submission file (be sure to pass hyperparameters that influence CNN structure)::

    $ ./make_submission.py debug12-all-ep4 sub_th0.4_debug12-all-ep4 --threshold 0.4

