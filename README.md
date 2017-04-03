# Lung-Cancer-Image-Classification
Classifying lung cancer using CT scans of the chest cavity.

So far I have built and run a baseline model. It has one CNN layer that is only
28 by 28 by 3, so it is still pretty small and simple. I also only have a little
over 1000 images split between the 5 classes downloaded so far and the smallest
class only has about 35 images.

I am preprocessing all of the images, doing things like shift the image left or
right, rotate it a few degrees, shrink and zoom them, flip them, etc. This will
help with both having a small dataset, although I will download more data, as
well as overfitting.
