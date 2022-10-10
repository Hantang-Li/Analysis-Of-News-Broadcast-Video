from charset_normalizer import detect
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
from time import time

def sift_keypoints(img):
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_keypoints(keypoints, descriptors, keypoints2, descriptors2, draw=True):
    matches = []
    for count, pts in enumerate(keypoints):
        f = descriptors[count]
        dists = []
        for count2, pts2 in enumerate(keypoints2):
            f2 = descriptors2[count2]
            dists.append(np.linalg.norm(f-f2))
        sorted_dists = np.argsort(np.array(dists))
        ratio = dists[sorted_dists[0]] / dists[sorted_dists[1]]
        if ratio < 0.75:
            matches.append((pts, keypoints2[sorted_dists[0]]))

    return matches

def homography(keypoints):
    A = []
    for pts in keypoints:
        a = np.array([pts[0].pt[0], pts[0].pt[1], 1, 0,0,0,-pts[1].pt[0]*pts[0].pt[0], -pts[1].pt[0]*pts[0].pt[1], -pts[1].pt[0]])
        b = np.array([0,0,0, pts[0].pt[0], pts[0].pt[1], 1,-pts[1].pt[1]*pts[0].pt[0], -pts[1].pt[1]*pts[0].pt[1], -pts[1].pt[1]])
        A.append(a)
        A.append(b)
    A = np.stack(A, axis=0)
    w, v = LA.eig(A.T @ A)
    min = np.argmin(w)
    h = v[:,min]
    return h.reshape((3,3))

def RANSAC(keypoints, thresh, flann=False, kp1=None, kp2=None):
    H_opt = None
    max_count = 0
    matches = []
    if flann:
        for match in keypoints:
            matches.append((kp1[match[0].queryIdx], kp2[match[0].trainIdx]))
    else:
        matches = keypoints

    for _ in range(5000):
        # select 4 pairs of matches randomly
        import random
        batch = random.sample(matches, 4)
        # compute the homography transformation
        H = homography(batch)

        # project all matched points, find inliers
        counter = 0
        for pt in matches:
            coordx, coordy = pt[0].pt
            coord = np.array([coordx, coordy, 1.])
            new_coord = np.matmul(H, coord)
            new_coord = np.array([new_coord[0]/new_coord[2],new_coord[1]/new_coord[2]])
            diff = LA.norm(new_coord - np.array([pt[1].pt[0], pt[1].pt[1]]))
            if diff < thresh:
                counter += 1
        if counter > max_count:
            max_count = counter
            H_opt = H
    return H_opt

def detect_logo(img1, img2, method):
    # img1 = cv2.imread(ref_img, 0)
    h, w = img1.shape
    # img2 = cv2.imread(target_img)
    h2, w2, _ = img2.shape
    if h2 >= h and w2 >= w:
        result = img2.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift_keypoints(img1)
        kp2, des2 = sift_keypoints(img2)
        
        flann = False
        if method == "dist":
            matches = match_keypoints(kp1, des1, kp2, des2, draw=False)
        elif method == "flann":
            matches = []
            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
            try:
                matches = matcher.knnMatch(des1, des2, 2)
                good_matches = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append((m, n))
                matches = good_matches
                flann = True
            except cv2.error:
                print("no match")

        if len(matches)>3:
            H_opt = RANSAC(matches, 5.0, flann=flann, kp1=kp1, kp2=kp2)
            # pts = np.float32([[0,0], [h-1,0], [h-1, w-1], [0, w-1]]).reshape(-1,1,2)
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            if H_opt is not None:
                pointsOut = cv2.perspectiveTransform(pts, H_opt.real).squeeze()
                if anomaly_detection(pointsOut):
                    result = cv2.polylines(result,[np.int32(pointsOut)],True,255,2, cv2.LINE_AA)
        # cv2.imwrite(os.path.join(out_dir, i+out_name), result)
        return result
    else:
        # cv2.imwrite(os.path.join(out_dir, i+out_name), img2)
        return img2
    # cv2.imwrite(os.path.join(out_dir, out_name), result)

def anomaly_detection(pts):
    if np.any(pts < 0):
        return False
    else:
        pt1 = pts[0]
        pt2 = pts[1]
        pt3 = pts[2]
        pt4 = pts[3]
        if pt1[1] > pt2[1] or pt2[0] > pt3[0] or pt3[1] < pt4[1] or pt4[0] < pt1[0]:
            return False
    return True

# This mini concat function is made by https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


if __name__ == "__main__":
    path = "clip_2_results"
    out_path = "clip_2_results"
    start_time = time()
    for i, fname in enumerate(os.listdir(path)):
        if ".jpg" in fname:
            img1 = cv2.imread("nbc_clip2.jpg", 0)
            img2 = cv2.imread(os.path.join(path, fname))
            h2, w2, _ = img2.shape
            imgs = []
            for i, r in enumerate(range(0,img2.shape[0],int(h2/3))):
                ar = []
                for j, c in enumerate(range(0,img2.shape[1],int(w2/3))):
                    k = detect_logo(img1, img2[r:r+int(h2/3), c:c+int(w2/3),:], "flann")
                    ar.append(k)
                imgs.append(ar)
            im_tile = concat_tile(imgs)
            cv2.imwrite(os.path.join(out_path, fname), im_tile)
    end_time = time()
    print(end_time - start_time)



