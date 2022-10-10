import numpy as np
import cv2
# import matplotlib.pyplot as plt
from numpy import linalg as LA
import os

# 0.3 for gradient, 
# 0.69 for non-gradient
THRESHOLD = 0.69

EDGE_DETECTION = False

if __name__ == "__main__":
    from time import time
    start_time = time()
    path = "clip_2_results"
    out_path = "clip_2_results"
    for fname in os.listdir(path):
        if ".jpg" in fname:
            img1 = cv2.imread("logo_clevver.jpg")
            w,h,_ = img1.shape
            img2 = cv2.imread(os.path.join(path, fname))
            result = img2.copy()

            if EDGE_DETECTION:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                edges_2 = cv2.Canny(image=img2, threshold1=100, threshold2=200)
                edges_1 = cv2.Canny(image=img1, threshold1=100, threshold2=200)
            
            detections = []
            template_matching = cv2.matchTemplate(
                img1, img2, cv2.TM_CCOEFF_NORMED
            )

            match_locations = np.where(template_matching >= THRESHOLD)

            for (x, y) in zip(match_locations[1], match_locations[0]):
                pts = np.array([[x,y], [x+h, y], [x+h,y+w], [x,y+w]])
                cv2.polylines(result,np.int32([pts]),True,255,2, cv2.LINE_AA)

            cv2.imwrite(os.path.join(out_path, fname), result)
    end_time = time()
    print(end_time - start_time)




