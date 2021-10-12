from typing import List, Tuple, Union, Optional, Mapping

from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy, reverse

from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np

import dataclasses
import math

from django.views.generic import CreateView, DetailView, DeleteView
from mediapipe.framework.formats import landmark_pb2
import threading
# Create your views here.
from healthapp.forms import  HealthCustomForm
from healthapp.models import HealthCustom
import threading as trd
import pyttsx3


### camera code ####

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

CUSTOMIZED_POSE_CONNETIONS: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32)
]

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> \
    Union[None, Tuple[int, int]]:
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class VideoCamera(object):
    def __init__(self, EXER, REPEATS, SET):
        self.video = cv2.VideoCapture(0)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # count variables
        self.left_counter: int = 0
        self.right_counter: int = 0
        self.counter: int = 0
        self.stages: List[str] = ['','']

        # 사용자가 미리 정해 놓은 루틴 / 한 운동의 반복횟수 / 한 운동의 세트횟수
        # self.ROUTE: List[str] = ['아령들기', '스쿼트','팔굽혀펴기']

        ### 수정중 ####
        self.ROUTE:List[str] = EXER
        self.REPEATS:List[int]= REPEATS
        self.SET: int = SET
        ##############

        self.current_route: int = 0  # ROUTE[0] .. ROUTE[2] 순으로 사용
        self.current_set: int = 0

        # color
        self.WHITE_COLOR = (224, 224, 224)
        self.BLACK_COLOR = (0, 0, 0)
        self.RED_COLOR = (0, 0, 255)
        self.GREEN_COLOR = (0, 128, 0)
        self.BLUE_COLOR = (255, 0, 0)

        # threading
        self.engine = pyttsx3.init()  ###
        self.engine.setProperty('rate', 200)

        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)

        # 추가 init_params
        self.WIDTH = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.HEIGHT = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        self.video.release()

    # custom draw mark

    def customized_draw_landmarks(self, image: np.ndarray,
                                  landmark_list: landmark_pb2.NormalizedLandmarkList,
                                  connections: Optional[List[Tuple[int, int]]] = None,
                                  landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(
                                      color=RED_COLOR),
                                  connection_drawing_spec: Union[
                                      DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):
        if not landmark_list:
            return
        if image.shape[2] != _RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        # revised_list = landmark_list.landmark[11:] # 0~10번까지 얼굴 좌표
        for idx, landmark in zip(range(11, len(landmark_list.landmark)), landmark_list.landmark[11:]):
            if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        #             print('idx_to_coordinates[idx] = landmark_px', idx, landmark_px[0], landmark_px[1])
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                     f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    drawing_spec = connection_drawing_spec[connection] if isinstance(
                        connection_drawing_spec, Mapping) else connection_drawing_spec
                    cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color,
                             drawing_spec.thickness)
        # Draws landmark points after finishing the connection lines, which is
        # aesthetically better.
        if landmark_drawing_spec:
            for idx, landmark_px in idx_to_coordinates.items():
                drawing_spec = landmark_drawing_spec[idx] if isinstance(
                    landmark_drawing_spec, Mapping) else landmark_drawing_spec
                # White circle border
                circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
                cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
                # Fill color into the circle
                cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)

    # coor == coordinates: 좌표
    def calculate_angle(self, coor_fst: List[float], coor_scd: List[float], coor_trd: List[float]) -> float:
        shoulder = np.array(coor_fst)
        elbow = np.array(coor_scd)
        wrist = np.array(coor_trd)

        radius = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1],
                                                                                   shoulder[0] - elbow[0])
        angle = np.abs(radius * 180 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def say_counter(self, counter: int):
        # engine = pyttsx3.init()
        # engine.setProperty('rate', 200)
        #
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[0].id)
        #
        self.engine.say(f"{counter}")
        try:
            self.engine.runAndWait()
        except:
            pass

    def create_thread(self, counter: int):
        # 중복 생성에 유의할 것
        thread = trd.Thread(target=self.say_counter, name='Counter', args=(counter,))
        thread.daemon = True
        if not thread.is_alive():
            thread.start()

    def draw_status(self, img, counter: int, current_set: int, cur_fststage: str, cur_scdstage: str = None,
                    exer_type: str = None,
                    _repeat: bool = True, _stage: bool = True, _set: bool = True, _exercise: bool = True,
                    multi_stages: bool = True):
        cv2.rectangle(img=img, pt1=(0, 0), pt2=(1300, 70), color=(245, 117, 16), thickness=-1)

        if multi_stages:
            if _stage:  # Stage Data
                cv2.putText(img=img, text='Left Stage', org=(500, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=img, text=cur_scdstage, org=(500, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(img=img, text='Right Stage', org=(800, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=img, text=cur_fststage, org=(800, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)
        else:
            if _stage:  # Stage Data
                cv2.putText(img=img, text='Stage', org=(500, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=img, text=cur_fststage, org=(500, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)

        if _repeat:  # Repeat Data
            cv2.putText(img=img, text='Reps', org=(11, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img=img, text=str(counter), org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)

        if _set:  # Set Data
            cv2.putText(img=img, text='Sets', org=(101, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img=img, text=str(current_set), org=(101, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)

        if _exercise:  # Exercise Data
            cv2.putText(img=img, text='Exercise', org=(161, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img=img, text=exer_type, org=(161, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)

    def is_visiblities(self, img, first: float, second: float, third: float) -> bool:
        if first < _VISIBILITY_THRESHOLD or second < _VISIBILITY_THRESHOLD or third < _VISIBILITY_THRESHOLD:
            # cv2.rectangle(img=img, pt1=(100, int(self.HEIGHT / 2 - 50)), pt2=(1110, int(self.HEIGHT / 2 + 20)), color=self.BLACK_COLOR,
            #               thickness=-1)
            cv2.rectangle(img=img, pt1=(50, int(self.HEIGHT / 2 - 50)), pt2=(550, int(self.HEIGHT / 2 + 20)), color=self.BLACK_COLOR,
                          thickness=-1)

            text = 'Please adjust your web camera'
            cv2.putText(img=img, text=text, org=(100, int(self.HEIGHT / 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)
            return True
        else:
            return False

    # def draw_end(img):
    #     text = 'Finally Finish You Did'
    #     cv2.putText(img=img, text=text, org=(100, int(HEIGHT / 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=2, color=WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)


    def get_frame(self):
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            checkframe, frame = self.video.read()
            # frame = cv2.flip(src=frame, flipCode=1) # 좌우(1) 또는 상하(0) 반전

            # Recolor 'frame' to RGB
            image = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make Detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                print('self.ROUTE[self.current_route]', self.ROUTE[self.current_route])

                # 루틴별로 연산 필요 -------------------------------------------------------
                # Get coordinates & Calculate a angle & Counter Logic
                if self.ROUTE[self.current_route] == 'Push Up':
                    if self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                                      second=landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                                      third=landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility) and \
                       self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                                      second=landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                                      third=landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility):
                        pass
                    else:
                        # 어깨, 팔꿈치, 손목
                        coor_left_shoulder: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        coor_left_elbow: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        coor_left_wrist: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        coor_right_shoulder: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        coor_right_elbow: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        coor_right_wrist: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        left_angle: float = self.calculate_angle(coor_left_shoulder, coor_left_elbow, coor_left_wrist)
                        right_angle: float = self.calculate_angle(coor_right_shoulder, coor_right_elbow, coor_right_wrist)

                        # 양쪽이 안보일 때는 어떻게 카운팅되는지 확인해보기
                        if left_angle >= 170 and right_angle >= 170 and self.stages[0] == 'DOWN':
                            self.stages[0] = 'UP'
                            self.counter += 1
                            self.create_thread(self.counter)
                        if left_angle <= 100 and right_angle <= 100:
                            self.stages[0] = 'DOWN'

                        if self.counter == self.REPEATS[self.ROUTE.index('Push Up')]:
                            if self.ROUTE[-1] == 'Push Up':
                                self.current_set += 1
                                if self.current_set == self.SET:
                                    pass
                            else:
                                self.current_route += 1
                                self.stages[0] = ''
                            self.counter = 0

                        self.draw_status(img=image, counter=self.counter , current_set=self.current_set, cur_fststage=self.stages[0], cur_scdstage=None,
                                    exer_type=self.ROUTE[self.current_route], _repeat=True, _stage=True, _set=True, _exercise=True, multi_stages=False)

                elif self.ROUTE[self.current_route] == 'Curl':
                    if self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                                      second=landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                                      third=landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility) or\
                        self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                                       second=landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                                       third=landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility):
                        pass
                    else:
                        # 어깨, 팔꿈치, 손목
                        coor_left_shoulder: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        coor_left_elbow: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        coor_left_wrist: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        coor_right_shoulder: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        coor_right_elbow: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        coor_right_wrist: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        left_angle: float = self.calculate_angle(coor_left_shoulder, coor_left_elbow, coor_left_wrist)
                        right_angle: float = self.calculate_angle(coor_right_shoulder, coor_right_elbow, coor_right_wrist)

                        if left_angle > 160:
                            self.stages[0] = 'DOWN'
                        elif left_angle < 30 and self.stages[0] == 'DOWN':
                            self.stages[0] = 'UP'
                            self.left_counter += 1

                        if right_angle > 160:
                            self.stages[1] = 'DOWN'
                        elif right_angle < 30 and self.stages[1] == 'DOWN':
                            self.stages[1] = 'UP'
                            self.right_counter += 1

                        if self.left_counter >= 1 and self.right_counter >= 1:
                            self.counter += 1
                            self.create_thread(self.counter)
                            self.left_counter = 0
                            self.right_counter = 0

                            if self.counter == self.REPEATS[self.ROUTE.index('Curl')]:
                                if self.ROUTE[-1] == 'Curl':
                                    self.current_set += 1
                                    if self.current_set == self.SET:
                                        pass
                                else:
                                    self.current_route += 1
                                    self.stages[0], self.stages[1] = '', ''
                                self.counter = 0

                        # Setup status box / Draw Status
                        self.draw_status(img=image, counter=self.counter, current_set=self.current_set, cur_fststage=self.stages[0], cur_scdstage=self.stages[1],
                                    exer_type=self.ROUTE[self.current_route], _repeat=True, _stage=True, _set=True, _exercise=True, multi_stages=True)

                elif self.ROUTE[self.current_route] == 'Squat':
                    if self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                                      second=landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
                                      third=landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility) or \
                       self.is_visiblities(img=image, first=landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                                      second=landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
                                      third=landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility):
                        pass
                    else:
                        # 골반(hip), 무릎(knee), 발목(ankle)
                        coor_left_hip: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        coor_left_knee: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        coor_left_ankle: List[float] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        coor_right_hip: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        coor_right_knee: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        coor_right_ankle: List[float] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        left_angle: float = self.calculate_angle(coor_left_hip, coor_left_knee, coor_left_ankle)
                        right_angle: float = self.calculate_angle(coor_right_hip, coor_right_knee, coor_right_ankle)

                        if left_angle >= 170 and right_angle >= 170 and self.stages[0] == 'DOWN':
                            self.stages[0] = 'UP'
                            self.counter += 1
                            self.create_thread(self.counter)
                        if left_angle <= 120 and right_angle <= 120:
                            self.stages[0] = 'DOWN'

                        if self.counter == self.REPEATS[self.ROUTE.index('Squat')]:
                            if self.ROUTE[-1] == 'Squat':
                                self.current_set += 1
                                if self.current_set == self.SET:
                                    pass
                            else:
                                self.current_route += 1
                                self.stages[0] = ''
                            self.counter = 0

                        self.draw_status(img=image, counter=self.counter, current_set=self.current_set, cur_fststage=self.stages[0], cur_scdstage=None,
                                    exer_type=self.ROUTE[self.current_route], _repeat=True, _stage=True, _set=True, _exercise=True, multi_stages=False)


                elif self.ROUTE[self.current_route] == 'End':
                    cv2.rectangle(img=image, pt1=(100, int(self.HEIGHT / 2 - 50)), pt2=(550, int(self.HEIGHT / 2 + 20)),
                                  color=self.BLACK_COLOR,
                                  thickness=-1)

                    text = 'Success!'
                    cv2.putText(img=image, text=text, org=(230, int(self.HEIGHT / 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=self.WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)
                # elif ROUTE[current_route] == 'something_2':
                #     pass
                # ------------------------------------------------------------------------
            except:
                pass

            self.customized_draw_landmarks(image=image,
                                      landmark_list=results.pose_landmarks,  # same type : landmarks[32]
                                      connections=CUSTOMIZED_POSE_CONNETIONS)  # mp_pose.POSE_CONNECTIONS

            # cv2.imshow('Mediapipe Feed', image)
             # image = self.frame
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()


    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# @gzip.gzip_page
# def detectme(request):
#     try:
#         cam = VideoCamera()
#         return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace;boundary=frame')
#     except:
#         print("에러입니다")
#         pass

# detectme 수정중
@gzip.gzip_page
def detectme(request,pk):
    try:
        # health = Health.objects.get(id=pk)
        health = HealthCustom.objects.get(id=pk)
        exer_list = [health.exercise_1,health.exercise_2 ,health.exercise_3 ]
        reps_list = [health.repeats_1,health.repeats_2, health.repeats_3]
        print('exer_list',exer_list)
        print('reps_list',reps_list)

        cam = VideoCamera(EXER=exer_list , REPEATS=reps_list ,SET=health.set)
        return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace;boundary=frame')
    except:
        print("에러입니다")
        pass

# custom page View


class HealthDeleteView(DeleteView):
    model = HealthCustom
    context_object_name = 'target_health'
    success_url = reverse_lazy('healthmapp:health')

    # def get_success_url(self):
    #     return reverse('healthmapp:health')


class TrainingView(DetailView):
    model = HealthCustom
    context_object_name = 'target_health'
    template_name = 'healthapp/training.html'


class HealthCompleteView(DetailView):
    model = HealthCustom
    context_object_name = 'target_health'
    template_name = 'healthapp/complete.html'


class HealthCustomCreationView(CreateView):
    model = HealthCustom
    form_class = HealthCustomForm
    success_url = reverse_lazy('healthmapp:training')
    template_name = 'healthapp/healthcustom.html'

    def get_success_url(self):
        return reverse('healthmapp:training', args=[self.object.pk])
