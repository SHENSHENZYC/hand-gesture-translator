# Hand Gesture to Sign Language Alphabet Translator

This project aims to develop a tool that translates hand gestures into sign language alphabet in real time on a web cam. The tool provides two different approaches for translation:

1. **CNN (Convolutional Neural Network)**: This approach involves cropping the hand portion of an image and using a pre-trained CNN model to classify the cropped image.

2. **Mediapipe Landmarks**: This approach utilizes Google's Mediapipe package to identify landmarks for the hand in each image. The coordinates of these hand landmarks are then used for classification.

## Requirements

- Python 3.9
- OpenCV
- PyTorch
- Mediapipe
- NumPy
- Scikit-Learn

## Installation

1. Clone this repository:

```bash
git clone https://github.com/SHENSHENZYC/hand-gesture-translator.git
```

2. Forward to the project directory:

```bash
cd hand-gesture-translator
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the tool, follow these steps:

1. Run the `livetest.py` script with the desired method (`landmark` or `cnn`):

```bash
python livetest.py --method landmark
```

or

```bash
python livetest.py --method cnn
```

2. The script will start capturing live video from your webcam and display the predictions for your hand gestures on the screen.

## Dataset and Privacy

The raw images used for training the models were captured on the web cam of the developer's local machine. In terms of privacy and data protection, the original dataset will not be publicized or shared with any third parties. The models have been trained on this private dataset to ensure accuracy and effectiveness.

We are committed to upholding user privacy and maintaining the confidentiality of any data used in this project. No personally identifiable information or sensitive data is collected or stored during the translation process. The tool solely focuses on real-time hand gesture recognition and does not transmit any data to external servers.

If you have any concerns regarding privacy or data usage, please feel free to reach out to us at [alexyczhao@gmail.com](mailto:alexyczhao@gmail.com) We value your privacy and will address any questions or inquiries promptly.

Note: The dataset used for training is not included in this repository. Only the trained models and the necessary code for running the translation tool are provided.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

Please ensure that your code follows the existing style conventions and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more information.

## Acknowledgements

- The hand landmarks detection and tracking is powered by Google's Mediapipe package (link to Mediapipe).
- Special thanks to the contributors of the open-source libraries used in this project.

## Contact

For any questions or suggestions, please feel free to reach out to me at [alexyczhao@gmail.com](mailto:alexyczhao@gmail.com)
