from skimage.segmentation import slic
from pydnet import *
from utils import *

import tensorflow as tf
import os
import random as rng
import cv2
import imutils
import numpy as np

# forces tensorflow to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

rng.seed(12345)

cap = cv2.VideoCapture('00001.mp4')

_, background = cap.read()
background = imutils.resize(background, width=min(512, background.shape[1]))
background = cv2.cvtColor(background, cv2.COLOR_BGR2Lab)

# backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
# backSub = cv2.bgsegm.createBackgroundSubtractorKNN()

def main(_):
    with tf.Graph().as_default():
        placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())

        loader = tf.train.Saver()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            loader.restore(sess, 'checkpoint/IROS18/pydnet')
            while True:
                ret, frame = cap.read()
                if frame is None:
                    break

                frame = imutils.resize(frame, width=min(512, frame.shape[1]))
                orig = frame.copy()

                # background subtraction
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

                mask = cv2.absdiff(frame, background)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # mask = backSub.apply(frame)

                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                ret, mask = cv2.threshold(mask, 4, 255, cv2.THRESH_BINARY)

                # mask pre-processing
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48)))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48)))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48)))

                """
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((12, 6),np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((12, 12),np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((24, 24),np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 8),np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(32,32)))
                """

                """
                # canny edge
                canny_output = cv2.Canny(mask, 64, 64 * 2)
                contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # convex hull
                hull_list = []
                for i in range(len(contours)):
                    hull = cv2.convexHull(contours[i])
                    hull_list.append(hull)
                
                # draw convex hull
                cv2.fillPoly(mask, hull_list, color=(255, 255, 255))
                """

                # depth estimation
                frame = cv2.resize(orig, (512, 256)).astype(np.float32) / 255.
                frame = np.expand_dims(frame, 0)
                disp = sess.run(model.results[0], feed_dict={placeholders['im0']: frame})
                disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')

                # masking
                disp_color = imutils.resize(disp_color, width=min(512, disp_color.shape[1]))

                mask = np.float32(cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR))/255.
                mask = cv2.resize(mask, (512, 256)).astype(np.float32)
                frame = cv2.bitwise_and(disp_color, mask)

                """
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                frame = cv2.bitwise_and(orig, mask)
                """

                cv2.imshow("Video", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                del frame
                del disp

            cap.release()
            cv2.destroyAllWindows()
     
if __name__ == '__main__':
    tf.app.run()
