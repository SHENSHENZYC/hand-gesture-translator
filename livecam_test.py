import os
import cv2
import pickle
import argparse
import numpy as np
import torch

from torch.nn import functional as F

from landmark_approach_src.create_dataset import set_hand_landmark_detector
from cnn_approach_src.train_classifier import HandClassifier, label_idx_converter
from cnn_approach_src.crop_imgs import make_annotation_box, make_bbox
    

def _landmark_predict():
    """Predict hand gestures using hand landmarks."""
    K = 3

    # load model
    with open('model/best_landmark_clf.pkl', 'rb') as f:
        model = pickle.load(f)

    ## present hand landmarks and sign alphabet predictions on live cam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: web cam access failed. Please adjust web cam accessibility and try again.")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles, hands = set_hand_landmark_detector()

    while True:
        success, frame = cap.read()
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # generate hand landmarks for web cam
        live_landmarks = hands.process(frame_rgb)
        if live_landmarks.multi_hand_landmarks:     # there could be more than one hand in live cam
            for hand_landmarks in live_landmarks.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmark_coords_ls, x1, x2, y1, y2 = make_annotation_box(hand_landmarks, h, w)

                # make prediction
                pred_proba = model.predict_proba([np.array(landmark_coords_ls)])[0]
                class_names = model.classes_
                topk_proba = pred_proba[np.argsort(pred_proba)][::-1][:K]
                topk_class_ascii = class_names[np.argsort(pred_proba)][::-1][:K]
                topk_class = [chr(class_ascii) for class_ascii in topk_class_ascii]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, ' | '.join([f'{topk_class[i]}: {topk_proba[i]*100:.2f}%'for i in range(K)]),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 200), 3,
                            cv2.LINE_AA)

        cv2.putText(frame, 'Press Q to exit', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _cnn_predict():
    K = 3

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: web cam access failed. Please adjust web cam accessibility and try again.")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles, hands = set_hand_landmark_detector()

    with open('data/hand_imgs.pkl', 'rb') as f:
        hand_data = pickle.load(f)
    num_classes, label_to_idx, idx_to_label = label_idx_converter(hand_data['labels'])
    out_size = hand_data['imgs'][0].shape[0]

    # load model
    model = HandClassifier(num_classes)
    model.load_state_dict(torch.load('model/cnn_clf.pth'))
    model.eval()

    while True:
        success, frame = cap.read()
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # generate hand landmarks for web cam
        live_landmarks = hands.process(frame_rgb)
        if live_landmarks.multi_hand_landmarks:     # there could be more than one hand in live cam
            for hand_landmarks in live_landmarks.multi_hand_landmarks:
                x1, x2, y1, y2 = make_bbox(hand_landmarks, h, w)
                hand_rgb = frame_rgb[max(y1, 0):min(y2+1, h-1), max(x1, 0):min(x2+1, w-1), :]
                hand_bw = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2GRAY)
                hand_bw_resized = cv2.resize(hand_bw, dsize=(out_size, out_size), interpolation=cv2.INTER_CUBIC)

                hand_bw_normalized = np.array(hand_bw_resized, dtype=np.float32) / 255
                img_input = torch.from_numpy(hand_bw_normalized.reshape(-1, 1, out_size, out_size))
                with torch.no_grad():
                    pred = model(img_input)[0]
                    pred_proba = F.softmax(pred, dim=0).detach().numpy()
                    topk_proba = pred_proba[np.argsort(pred_proba)][::-1][:K]
                    topk_class = [idx_to_label[idx] for idx in np.argsort(pred_proba)[::-1][:K]]

                _, x1_ann, x2_ann, y1_ann, y2_ann = make_annotation_box(hand_landmarks, h, w)
                cv2.rectangle(frame, (x1_ann, y1_ann), (x2_ann, y2_ann), (0, 0, 0), 4)
                cv2.putText(frame, ' | '.join([f'{topk_class[i]}: {topk_proba[i]*100:.2f}%'for i in range(K)]),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 200), 3,
                            cv2.LINE_AA)

        cv2.putText(frame, 'Press Q to exit', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=['landmark', 'cnn'], help="Method to translate hand gesture to sign language MNIST: choose \"landmark\" or \"cnn\"")

    args = parser.parse_args()
    args_config = dict()
    for arg in vars(args):
        args_config[arg] = getattr(args, arg)
    method = args_config.get('method').strip().lower()

    if method == 'landmark':
        _landmark_predict()
    else:
        _cnn_predict()


if __name__ == "__main__":
    main()
