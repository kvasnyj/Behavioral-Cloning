Project: Behavioral Cloning
====================
1. Augmentation
---------------
8000+ lines in Udacity data set is not enough for a full train, especially for recovery and for generalization for track 2. I applied following augmentation technics: 
1. Select camera from (left, center, right)
2. Randomly change brightness
3. Transition horizontally 
4. Crop, to reduce non-valued information
5. Random shadow, for the second track
6. Flip

In additional, I tried to draw on images previous value of steering (in fact - current direction),  but without success. 

2. Model
-----------------------
3. Hyperparameters
* batch_size = 512
* samples_per_epoch = 39936
* epochs = 10
---------------------------------
4. Result
---------------------------------------
Totally I make 40 runs with different model architectures, different augmentations, and different hyperparameters. The final model passed track 1 and half of track 2. 
The problem with track 2 is the sharp turn to the right on the descent after the tunnel: set throttle to 0 is not enought to decrease speed value, I need a brake to pass it. 
