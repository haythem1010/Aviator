import json
from pprint import pprint

import pytesseract as ts
import os
import pyautogui as pt
import time as t
from datetime import datetime
from PIL import Image
from utils.helpers import keep_numeric
import cv2
import numpy as np

class Recorder(object):
    screen_shots_path = f"C:/Users/hayth/Desktop/screenshots/"
    path_to_ts = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    def capture_screen(self, x, y, width, height, data_nature, save=False):
        global save_path
        screenshot = pt.screenshot(region=(x, y, width, height))
        resized_screenshot = screenshot.resize((width * 2, height * 2), resample=Image.LANCZOS)
        timestamp = datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        if save:
            save_path = os.path.join(self.screen_shots_path, f"{data_nature}_screenshot_{timestamp}.png")
            resized_screenshot.save(save_path)
        return resized_screenshot, save_path, timestamp

    def read_my_screenshots(self, file_name, screenshot, local_file=True):
        if local_file:
            file = f"{self.screen_shots_path}{file_name}"
        else:
            file = screenshot

        ts.pytesseract.tesseract_cmd = self.path_to_ts
        # text detecting
        text = ts.image_to_string(file)
        return text

    @staticmethod
    def detect_red_color(image_path):
        # Read the image
        image = cv2.imread(image_path)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define the lower and upper bounds for red color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Convert the image to HSV
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Create masks to isolate red color in both ranges
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

        # Combine the masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Check if any red pixels are detected
        has_red_color = np.any(mask)

        return has_red_color

    @staticmethod
    def data_collection(data, time, result, total_bet, cash_out):
        data.append({'date': time,
                     'result': result,
                     'total_bet': total_bet,
                     'cash_out': cash_out})
        return None

    @staticmethod
    def data_cleaning(data):
        test1 = False
        test2 = False
        test3 = False

        i = 1
        clean = []
        while i < len(data):
            if data[i]['cash_out'] == '' or data[i]['total_bet'] == '':
                test1 = False
            if data[i]['cash_out'] < data[i-1]['cash_out'] or data[i]['total_bet'] < data[i]['total_bet']:
                test2 = False
            if data[i]['result'] == '':
                test3 = False
            if test1 and test2 and test3:
                clean.append({'date': data[i]['time'],
                              'result': data[i]['result'],
                              'total_bet': data[i]['total_bet'],
                              'cash_out': data[i]['cash_out']})
            i += 1
        return clean

    def main(self, file):
        start = t.time()

        interval = 1
        counter = 1
        data = []
        bet = []
        cash = []
        try:
            while True:
                """ Screenshot Processing """
                result_jpg, save1, timestamp1 = self.capture_screen(790, 345, 320, 120, 'result', save=True)
                total_bet_jpg, save2, timestamp2 = self.capture_screen(510, 720, 70, 60, 'total_bet', save=True)
                total_cash_out_jpg, save3, timestamp3 = self.capture_screen(1320, 720, 70, 60, 'total_cash_out', save=True)
                result = self.read_my_screenshots(file, result_jpg, local_file=False)
                total_bet = self.read_my_screenshots(file, total_bet_jpg, local_file=False)
                total_cash_out = self.read_my_screenshots(file, total_cash_out_jpg, local_file=False)
                test_end_op = self.detect_red_color(save1)

                bet.append(keep_numeric(total_bet))
                cash.append(keep_numeric(total_cash_out))

                if test_end_op:
                    """ Results """
                    result = keep_numeric(result)
                    total_bet = keep_numeric(total_bet)
                    total_cash_out = keep_numeric(total_cash_out)
                    print("Result  extracted from the photo : ", result)
                    print("Total bet  extracted from the photo : ", bet[counter-2])
                    print("Total cash out extracted from the photo : ", cash[counter-2])
                    print("--------------------------------------------------")
                    self.data_collection(data, time=timestamp3, result=result, total_bet=bet[counter-2], cash_out=cash[counter-2] )
                counter += 1
                t.sleep(interval)
        except KeyboardInterrupt:
            print("Screenshots capturing stopped ")

        # clean = self.data_cleaning(data_1)
        # pprint(clean)
        with open("data_2", 'w') as json_file:
            json.dump(data, json_file)

        print(bet)
        print(cash)
        """ Execution Time : """
        end = t.time()
        print(round(end - start, 3), "sec")
        return result, total_bet, total_cash_out


text = Recorder()
# Readers.tile('1.png', 'C:/Users/hayth/Desktop/screenshots', 'C:/Users/hayth/Desktop/screenshots/1', 120)
text.main('1.png')
