import cv2
import numpy as np
import copy

threshold = 65
blurvalue = 15
subthreshold = 45


def get_max_contour(contours):
    # Returns the contour with maximum area
    length = len(contours)
    maxArea = -1
    ci = - 1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
    return ci


cap = cv2.VideoCapture(0)
bg_model = cv2.createBackgroundSubtractorKNN(0, subthreshold, detectShadows=False)

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    foreground_mask = bg_model.apply(frame)

    res = cv2.bitwise_and(frame, frame, mask=foreground_mask)

    blur = cv2.GaussianBlur(res, (blurvalue, blurvalue), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # contour Detection
    thresh1 = copy.deepcopy(thresh)

    im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find Max Contour
    try:
        max_index = get_max_contour(contours)
        res = contours[max_index]
        hull = cv2.convexHull(res)

        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)
        cv2.drawContours(frame, [res], -1, (255, 150, 0), 3)
    except IndexError:
        print("No Contours Found")

    # Finding convexity defects
    cnt = contours[get_max_contour(contours)]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    min_dimension = 0
    max_dimension = 0
    i = 0

    for i in range(defects.shape[0]):
        s, e, f, g = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        dist = cv2.pointPolygonTest(cnt, centr, True)
        cv2.line(frame, start, end, [0, 255, 0], 2)
        cv2.circle(frame, far, 5, [0, 0, 255], -1)
        print(i)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('Frame', 1024, 768)

    cv2.imshow('Frame', frame)

    if (cv2.waitKey(1) == 13):
        break

cap.release()
cv2.destroyAllWindows()
