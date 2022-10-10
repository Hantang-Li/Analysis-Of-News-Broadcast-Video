# Analysis-Of-News-Broadcast-Video
Computer Vision Video Analysis Project

Authors: Hantang Li, Hongyi Sun
----------------------------------------------------------------
To produce the final result, please follow the following steps:

1. Run notebook file [detection.ipynb]
2. At the end, download the .zip file generated by the code
The .zip file contains frames with detected faces/gender
Now we use those frames to detect logo.
3. For clip 1 run [logo_template_match.py] only, and use it to
detect "logo2.png".
4. For clip 2, run [logo_template_match.py] first to detect
"logo_clevver.jpg", then run [logo_feature_match.py] to detect
"nbc_clip2.jpg"
5. For clip 3, run [logo_template_match.py] only, and use it to
detect "logo_flick.jpg".
----------------------------------------------------------------

detection.ipynb:
This file contains codes for part (f), (g) and (i).
By running all the code cells (except for the ones I noted \not 
to run), you will generate a .zip file for processed frames for 
Clip 2.

If you want to detect other clip, simply change all occurence in 
the code of "clip 2" to "clip x". Where x in [1, 2, 3]

----------------------------------------------------------------

train_face_detection.ipynb:
This file contains codes to train the face detector using Yolov5
+ CelebA dataset. Note: Downloading CelebA dataset and extracting
can take up to 20 hrs. Training Yolov5 medium on 70k dataset size 
takes roughly 50 min per epoch on p100 GPU.

----------------------------------------------------------------

logo_template_match/logo_feature_match.py:
These 2 files contains codes to detect logo using template and 
feature matching. Please change the corresponding logo & image
path to your local path.

----------------------------------------------------------------
