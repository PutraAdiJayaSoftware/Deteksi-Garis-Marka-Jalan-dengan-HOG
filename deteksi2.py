import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def region_of_interest_v1(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


dirData = "/home/pi/Project/UJI_ACAK/"#"/home/pi/Project/2/lowress/"#"/home/pi/Project/2/fix tengah marka/"##"/home/pi/Project/2/fix mendekati marka/" #"/home/pi/Project/2/fix tengah marka/"#"
arr = os.listdir(dirData)


def draw_lines_v1(height, width, img, lines, color=[255, 0, 0], thickness=5):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img


def draw_lines(height, width, img, lines, color=[255, 255, 255], thickness=3):
    if lines is None:
        return


    img = np.copy(img)

    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    cek = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cek += x1;


    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    jarak = abs(width/2 - x1)
    print (jarak)
    color2 = [0, 0, 255]
    if jarak<=160:
        color2 = [0, 0, 255]
        cek = 1
    elif jarak<=280:
        color2 = [0, 255, 255]
        cek = 1
    else:
        color2 = [0, 255, 0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    cv2.circle(img, (width / 2, height), 20, color2, -1)

    cv2.imshow('Video Frame', img)
    return img


def readVideo(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape
            region_of_interest_vertices = [
                (0, height),
                (0, height * 0.8),
                (width * 0.5, height * 0.6),
                (width, height * 0.8),
                (width, height),
            ]

            resized_image = frame
            image = resized_image
            blur = cv2.blur(image, (5, 5))
            gray_image = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.equalizeHist(gray_image)
            cannyed_image = cv2.Canny(gray_image, 100, 200)
            cropped_image = region_of_interest(
                cannyed_image,
                np.array([region_of_interest_vertices], np.int32),
            )

            lines = cv2.HoughLinesP(
                cropped_image,
                rho=6,
                theta=np.pi / 60,
                threshold=128,
                lines=np.array([]),
                minLineLength=40,
                maxLineGap=40
            )
            #print (lines)
            #line_image = draw_lines(image, lines)
            #cv2.imshow('cannyed_image',cannyed_image)

            left_line_x = []
            left_line_y = []
            if lines is None:
                continue

            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])

            min_y = int(height * 0.6)
            max_y = int(height)

            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))

            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))

            line_image = draw_lines(height, width,
                image,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                ]],
                thickness=5,
            )
            # cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1000) == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


for fi in arr:
    path = ("".join([dirData, fi]))
    print (path)
    readVideo(path)

