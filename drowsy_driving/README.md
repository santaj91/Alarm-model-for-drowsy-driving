# Alarm-model-for-drowsy-driving

Python = 3.7.13 , Tensorflow = 2.8.2

This model makes a sound when driver's eyes are closed for a few seconds. To make it precisely, i added a smile detection. so this algorithms could judge if driver's eyes are closed more precisely.

![drowsy_AdobeExpress (1)](https://user-images.githubusercontent.com/93965016/180709045-3df84649-b35e-4fb1-b203-1a9c7f0ce8e6.gif)


# Models

## Eye detector
At first, i trained only closing eye detector with eyes of Westerners. But i recognized that it didn't work when driver is asian. So i retrained with asian's eyes that i cropped from asian dataset with opencv. And than this model is able to detect asian's eye.

## Smile detector
Another challenge occured when driver smiles. smiling face can make human's eyes smaller than before. so this algorithms could malfunction. To protect this, i added smile detector. so when the algorithm detect a human smiling, a counter which can calculate how long driver's eyes are closed become a zero.

## Algorithm
A counter counts numbers continuously when the driver's eyes are closed. when counter become the number of ten, a Alram makes sound. If the driver opens eyes before the counter become the number of ten and the detector can found the driver smiling , the counter becomes zero.  

# Dataset

https://github.com/kairess/eye_blink_detector <br>
https://afad-dataset.github.io/ <br>
https://www.kaggle.com/datasets/ghousethanedar/smiledetection

# Reference
https://github.com/kairess/eye_blink_detector
