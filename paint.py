import mediapipe
import cv2

import numpy
import math

from tensorflow import lite


class load:
    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        """loads tflite model

        Args:
            model_path (str): path to model
            num_threads (int, optional): number of threads. Defaults to 1.
        """
        self.interpreter = lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def preprocess(landmarks: numpy.ndarray) -> numpy.ndarray:
        """normalising the landmarks

        Args:
            landmarks (numpy.ndarray): landmarks array

        Returns:
            numpy.ndarray: array of normalised points
        """
        x, y = (0, 0)

        for num, landmark in enumerate(landmarks):
            if num == 0:
                x, y = landmark

            landmarks[num][0] = landmarks[num][0] - x
            landmarks[num][1] = landmarks[num][1] - y

        landmarks = landmarks.flatten()

        return landmarks / abs(landmarks).max()

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        """make predition base on hand landmarks

        Args:
            data (numpy.ndarray): hand landmarks array

        Returns:
            numpy.ndarray: predictions array
        """
        tensor_index = self.input_details[0]["index"]

        self.interpreter.set_tensor(
            tensor_index, numpy.array([self.preprocess(data.copy())], numpy.float32)
        )
        self.interpreter.invoke()

        tensor_index = self.output_details[0]["index"]

        return self.interpreter.get_tensor(tensor_index)


def overlay(
    background: numpy.ndarray, foreground: numpy.ndarray, point: tuple[int]
) -> None:
    """overlay PNG image to another png image

    Args:
        background (numpy.ndarray): background image
        foreground (numpy.ndarray): foreground image
        point (tuple[int]): position of foreground image
    """
    FGX = max(0, point[0] * -1)
    BGX = max(0, point[0])
    BGY = max(0, point[1])
    FGY = max(0, point[1] * -1)

    BGH, BGW = background.shape[:2]
    FGH, FGW = foreground.shape[:2]

    W = min(FGW, BGW, FGW + point[0], BGW - point[0])
    H = min(FGH, BGH, FGH + point[1], BGH - point[1])

    foreground = foreground[FGY : FGY + H, FGX : FGX + W]
    backgroundSubSection = background[BGY : BGY + H, BGX : BGX + W]

    alphaMask = numpy.dstack(tuple(foreground[:, :, 3] / 255.0 for _ in range(3)))

    background[BGY : BGY + H, BGX : BGX + W] = (
        backgroundSubSection * (1 - alphaMask) + foreground[:, :, :3] * alphaMask
    )


drawing = mediapipe.solutions.drawing_utils
styles = mediapipe.solutions.drawing_styles

hands = mediapipe.solutions.hands

COLORS = {
    "cyan": (196, 177, 5),
    "white": (255, 255, 255),
    "green": (75, 161, 0),
    "red": (1, 1, 227),
    "orange": (0, 102, 255),
    "yellow": (6, 206, 254),
    "pink": (233, 71, 236),
    "gray": (118, 139, 130),
}

RADIUS = 34

OPTIONS = {
    "brush": cv2.imread(r"./assets/images/brush.png", cv2.IMREAD_UNCHANGED),
    "rectangle": cv2.imread(r"./assets/images/rectangle.png", cv2.IMREAD_UNCHANGED),
    "circle": cv2.imread(r"./assets/images/circle.png", cv2.IMREAD_UNCHANGED),
    "line": cv2.imread(r"./assets/images/line.png", cv2.IMREAD_UNCHANGED),
    "eraser": cv2.imread(r"./assets/images/eraser.png", cv2.IMREAD_UNCHANGED),
}

model = load(r"./assets/model/model.tflite")

currentColor = False
currentOption = False

landmarks = numpy.zeros((21, 2), numpy.int32)
pointOne = None

SHAPE = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, SHAPE[0])
cap.set(4, SHAPE[1])

mask = numpy.zeros_like(cap.read()[1])

with hands.Hands(
    max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3
) as detector:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        cv2.flip(frame, 1, frame)

        frame.flags.writeable = False

        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for num, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[num][0] = landmark.x * frame.shape[1]
                    landmarks[num][1] = landmark.y * frame.shape[0]

                prediction = numpy.argmax(model(landmarks))

        else:
            pointOne = None

        for num, (option, image) in enumerate(OPTIONS.items()):
            point = numpy.array([22, 16 + num * frame.shape[0] // 5])

            if results.multi_hand_landmarks:
                if not prediction and (
                    landmarks[8][0] >= point[0]
                    and landmarks[8][0] <= point[0] + image.shape[1]
                    and landmarks[8][1] >= point[1]
                    and landmarks[8][1] <= point[1] + image.shape[0]
                ):
                    currentOption = option

            if currentOption == option:
                cv2.rectangle(
                    frame, point, point + image.shape[-2::-1], (255, 100, 0), 4
                )

            overlay(frame, image, point)

        for num, (color, value) in enumerate(COLORS.items()):
            center = (
                frame.shape[1] - 80,
                frame.shape[0] // (len(COLORS) + 1) * (num + 1),
            )

            cv2.circle(frame, center, RADIUS, value, -1)

            if results.multi_hand_landmarks:
                if not prediction and (
                    landmarks[8][0] >= center[0] - RADIUS
                    and landmarks[8][0] <= center[0] + RADIUS
                    and landmarks[8][1] >= center[1] - RADIUS
                    and landmarks[8][1] <= center[1] + RADIUS
                ):
                    currentColor = color

            if currentColor == color:
                cv2.circle(frame, center, RADIUS, COLORS["white"], 4)

        if results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                hands.HAND_CONNECTIONS,
                styles.DrawingSpec((0, 0, 255), -1, 6),
                styles.DrawingSpec((0, 255, 0), 3, -1),
            )

            if prediction and currentColor and currentOption:
                if landmarks[8][0] > 160 and landmarks[8][0] < 1120:
                    if currentOption == "brush":
                        cv2.circle(mask, landmarks[8], 14, COLORS[currentColor], -1)

                    elif currentOption == "eraser":
                        cv2.circle(mask, landmarks[8], 36, 0, -1)
                        cv2.circle(frame, landmarks[8], 36, COLORS["white"], 6)
                        cv2.circle(frame, landmarks[8], 20, COLORS["white"], -1)

                    else:
                        if pointOne is None:
                            pointOne = landmarks[8].copy()

                        if (
                            abs(landmarks[8][0] - pointOne[0]) > 10
                            and abs(landmarks[8][1] - pointOne[1]) > 10
                            and currentOption == "rectangle"
                        ):
                            cv2.rectangle(
                                frame, pointOne, landmarks[8], COLORS[currentColor], 5
                            )

                        elif currentOption == "circle":
                            centerOFCircle = (pointOne + landmarks[8]) // 2

                            cv2.circle(
                                frame,
                                centerOFCircle,
                                int(math.hypot(*(centerOFCircle - pointOne))),
                                COLORS[currentColor],
                                5,
                            )

                        elif currentOption == "line":
                            cv2.line(
                                frame, pointOne, landmarks[8], COLORS[currentColor], 5
                            )

            elif (
                not prediction
                and currentColor
                and currentOption
                and pointOne is not None
            ):
                if currentOption == "rectangle":
                    cv2.rectangle(mask, pointOne, landmarks[8], COLORS[currentColor], 5)

                elif currentOption == "circle":
                    cv2.circle(
                        mask,
                        centerOFCircle,
                        int(math.hypot(*(centerOFCircle - pointOne))),
                        COLORS[currentColor],
                        5,
                    )

                elif currentOption == "line":
                    cv2.line(mask, pointOne, landmarks[8], COLORS[currentColor], 5)

                pointOne = None

        _, thresholdMask = cv2.threshold(
            cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY_INV
        )

        finalMask = cv2.bitwise_and(
            frame, cv2.cvtColor(thresholdMask, cv2.COLOR_GRAY2BGR)
        )

        cv2.imshow("Paint 3D", cv2.add(finalMask, mask))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()

cv2.destroyAllWindows()
