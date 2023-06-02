import os
import cv2
import pickle
import mediapipe as mp


def set_hand_landmark_detector():
    """Set up hand landmark detector."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    return mp_hands, mp_drawing, mp_drawing_styles, hands


def main():
    """Entry point of the program."""
    ROOT = '.' if os.path.basename(os.getcwd()) == 'my_project' else '..'
    DATA_DIR = os.path.join(ROOT, 'data/raw')

    # initialize hand landmark detector object
    mp_hands, mp_drawing, mp_drawing_styles, hands = set_hand_landmark_detector()

    data = []
    labels = []
    img_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for name in files:
            if not name.endswith('.jpg'):
                continue

            img = cv2.imread(os.path.join(root, name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detect landmarks on hand
            landmarks = hands.process(img_rgb)
            landmark_coords_ls = []
            landmark_xs = []
            landmark_ys = []
            if len(landmarks.multi_hand_landmarks) == 1:  # make sure exactly one hand is detected
                for hand_landmarks in landmarks.multi_hand_landmarks:
                    for landmark_coords in hand_landmarks.landmark:
                        landmark_xs.append(landmark_coords.x)
                        landmark_ys.append(landmark_coords.y)
                    for landmark_coords in hand_landmarks.landmark:
                        landmark_coords_ls.append(landmark_coords.x - min(landmark_xs))
                        landmark_coords_ls.append(landmark_coords.y - min(landmark_ys))

            data.append(landmark_coords_ls)
            labels.append(os.path.join(root, name).split('/')[-2].strip())
            img_paths.append(os.path.join(root, name))

    with open(os.path.join(ROOT, 'data/landmarks.pkl'), 'wb') as f:
        pickle.dump({'data': data, 'labels': labels, 'img_paths': img_paths}, f)


if __name__ == '__main__':
    main()
