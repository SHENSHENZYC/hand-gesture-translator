import os
import cv2
import argparse


def main():
    """Entry point of the program."""
    # create a data folder to store images for training model
    DATA_DIR = 'data/raw'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # modifiable arguments that controls configuration of the dataset being created
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=24)
    parser.add_argument('--class_names', type=str, default='abcdefghiklmnopqrstuvwxy')
    parser.add_argument('--dataset_size', type=int, default=100)

    args = parser.parse_args()
    args_config = dict()
    for arg in vars(args):
        args_config[arg] = getattr(args, arg)

    if args_config.get('num_classes') \
        != len(args_config.get('class_names')):
        print("ERROR: number of class names must match number of classes.")
        return

    class_names = args_config.get('class_names')

    # web cam access
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: web cam access failed. Please adjust web cam accessibility and try again.")
        return

    for i in range(args_config.get('num_classes')):
        if not os.path.exists(os.path.join(DATA_DIR, class_names[i])):
            os.makedirs(os.path.join(DATA_DIR, class_names[i]))
        
        print(f"Collecting data for class '{class_names[i]}' ...")

        # set video capturing starting key
        while True:
            success, frame = cap.read()
            cv2.putText(frame, f'Ready? Press Q to generate images for "{class_names[i]}"',
                        (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # capture 100 video screenshots every 25 milliseconds
        cnt = 0
        while cnt < args_config.get('dataset_size'):
            success, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, class_names[i], f'{cnt}.jpg'), frame)
            cnt += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

