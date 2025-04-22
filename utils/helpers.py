from datetime import datetime, timedelta
from PIL import Image
from itertools import product
import os
import cv2
import numpy as np


def remove_extension(file_path):
    name = file_name_without_extension = file_path.rsplit('.', 1)[0]
    return name
#------------------------------------------------------------------------------------------------------

def delete_duplicates(list):
    my_set = set()
    list_no_dp = [i for i in list if not(i in my_set or my_set.add(i))]
    return list_no_dp
#------------------------------------------------------------------------------------------------------

def convert_timestamp_to_date(time, output_format='%Y-%m-%d %H:%M:%S'):
    dt = datetime.fromtimestamp(time)
    return dt.strftime(output_format)
#------------------------------------------------------------------------------------------------------

def create_time_range(start_date, end_date, delay):
    time_range = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').date()
    while start_date <= end_date:
        time_range.append(start_date)
        start_date = start_date + timedelta(minutes=delay)

    return time_range
#------------------------------------------------------------------------------------------------------

def decompose_image_into_tiles(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    print(w, ",", h)

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
    return 0

#------------------------------------------------------------------------------------------------------

def remove_background(screen_shots_path, save_path):
    # Read the image
    image = cv2.imread(screen_shots_path+'screenshot_2023_12_10 20_22_42.png',cv2.IMREAD_UNCHANGED)
    print("Image Shape:", image.shape)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([200, 200, 200], dtype=np.uint8)
    # Create a mask for white pixels
    mask_white = cv2.inRange(image, lower_white, upper_white)

    # Create an inverted mask to keep white pixels
    # Convert the image to grayscale
    mask_inv = cv2.bitwise_not(mask_white)

    # Create a black background
    black_background = np.zeros_like(image, np.uint8)

    # Copy the image from the original screenshot to the black background
    result = cv2.bitwise_or(black_background, image, mask=mask_inv)
    negative_result = cv2.bitwise_not(result)

    # Resize the image to the desired dimensions
    resized_result = cv2.resize(negative_result, (140, 120), interpolation=cv2.INTER_AREA)

    # Save the result as a JPEG image
    cv2.imwrite(save_path, resized_result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

#------------------------------------------------------------------------------------------------------

def keep_numeric(string):
    numeric = ''
    for i in string:
        if '0' <= i <= '9' or i == '.':
            numeric += i
    return numeric

#------------------------------------------------------------------------------------------------------
def detect_red_color(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Convert the image to HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Create a mask to isolate red color
    mask = cv2.inRange(image_hsv, lower_red, upper_red)

    # Check if any red pixels are detected
    has_red_color = np.any(mask)

    return has_red_color