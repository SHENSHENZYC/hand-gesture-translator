import os
import cv2
import pickle
import argparse
import mediapipe as mp
import numpy as np


def make_annotation_box(hand_landmarks, h, w):
    landmark_coords_ls = []
    landmark_xs = []
    landmark_ys = []
    for landmark_coords in hand_landmarks.landmark:
        landmark_xs.append(landmark_coords.x)
        landmark_ys.append(landmark_coords.y)

    for landmark_coords in hand_landmarks.landmark:
        landmark_coords_ls.append(landmark_coords.x - min(landmark_xs))
        landmark_coords_ls.append(landmark_coords.y - min(landmark_ys))

    x1 = int(min(landmark_xs) * w) - 20
    y1 = int(min(landmark_ys) * h) - 20

    x2 = int(max(landmark_xs) * w) + 20
    y2 = int(max(landmark_ys) * h) + 20

    return landmark_coords_ls, x1, x2, y1, y2

def make_bbox(hand_landmarks, h, w):
    """Generate bounding box from hand landmarks."""
    _, x1, x2, y1, y2 = make_annotation_box(hand_landmarks, h, w)

    hand_width = np.abs(x1 - x2)
    hand_height = np.abs(y1 - y2)
    diff = np.abs(hand_width - hand_height)
    if hand_width < hand_height:
        x1 = int(x1 - diff/2)
        x2 = int(x2 + diff/2)
    else:
        y1 = int(y1 - diff/2)
        y2 = int(y2 + diff/2)

    return x1, x2, y1, y2


def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=28)
    args = parser.parse_args()
    args_config = dict()
    for arg in vars(args):
        args_config[arg] = getattr(args, arg)
    out_size = args_config.get('size')

    # set up hand detection tool
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    ROOT = '.' if os.path.basename(os.getcwd()) == 'my_project' else '..'
    DATA_DIR = os.path.join(ROOT, 'data/raw')

    imgs = []
    labels = []
    img_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for name in files:
            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, _ = img_rgb.shape
            class_name = img_path.split('/')[-2].strip()

            landmarks = hands.process(img_rgb)
            if len(landmarks.multi_hand_landmarks) == 1:  # make sure exactly one hand is detected 
                for hand_landmarks in landmarks.multi_hand_landmarks:
                    x1, x2, y1, y2 = make_bbox(hand_landmarks, h, w)
                    hand_rgb = img_rgb[max(y1, 0):min(y2+1, h-1), max(x1, 0):min(x2+1, w-1), :]
                    hand_bw = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2GRAY)
                    hand_bw_resized = cv2.resize(hand_bw, dsize=(out_size, out_size), interpolation=cv2.INTER_CUBIC)

            imgs.append(hand_bw_resized)
            labels.append(class_name)
            img_paths.append(img_path)

    with open(os.path.join(ROOT, 'data/hand_imgs.pkl'), 'wb') as f:
        pickle.dump({'imgs': imgs, 'labels': labels, 'img_paths': img_paths}, f)


if __name__ == '__main__':
    main()
