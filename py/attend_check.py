import copy

import dlib, cv2
import numpy as np
import os

class attend_check:
    descs = None
    detector = None
    sp = None
    facerec = None

    def __init__(self):
        self.descs = np.load('member_descs.npy', allow_pickle=True)[()]
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

    def set_decsc(self):
        self.descs = np.load('member_descs.npy', allow_pickle=True)[()]

    def set_model(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

    def find_faces(self, img):
        dets = self.detector(img, 1)

        # face Not Found empty 0 return
        if len(dets) == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        rests, shapes = [], []
        shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)

        for k, d in enumerate(dets):
            rect = ((d.left(), d.top()), (d.right(), d.bottom()))
            rests.append(rect)
            shape = self.sp(img, d)

            # convert dlib shape to numpy array
            for i in range(0, 68):
                 shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

            shapes.append(shape)
        return rests, shapes, shapes_np


    def encode_faces(self, img, shapes):
        face_descriptors = []
        for shape in shapes:
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            face_descriptors.append(np.array(face_descriptor))
        return np.array(face_descriptors)

    def encode_face(self, img):
        dets = self.detector(img, 0)

        if len(dets) == 0:
            return np.empty(0)

        for k, d in enumerate(dets):
            shape = self.sp(img, d)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)

            return np.array(face_descriptor)

    def make_descs(self):
        path = './memberImg_list'

        new_descs = {}
        img_paths = {}

        with os.scandir(path) as list:
            for data in list:
                if data.is_file():
                    item = data.name
                    member = item.split('.')
                    img_paths[member[0]] = path + '/' + item

        for name, img_paths in img_paths.items():
            img_bgr = cv2.imread(img_paths, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        _, img_shapes, _ = self.find_faces(img_rgb)

        print('img_shapes : ', img_shapes)

        new_descs[name] = self.encode_faces(img_rgb, img_shapes)[0]
        np.save('member_descs.npy', new_descs)
        print(new_descs)

    def read_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def detect_member(self, img):
        img_path = './attendImg/' + img # me
        img_ori = self.read_img(img_path)
        img_cv = copy.deepcopy(img_ori)

        dets = self.detector(img_cv, 1)

        for k, d in enumerate(dets):
            shape = self.sp(img_cv, d)
            face_descriptor = self.facerec.compute_face_descriptor(img_cv, shape)

            last_found = {'name': 'unknown', 'dist': 0.4, 'color': (0, 0, 255)}

            for name, saved_desc in self.descs.items():
                dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

                if dist < last_found['dist']:
                    last_found = {'name': name, 'dist': dist, 'color': (255, 255, 255)}

            cv2.rectangle(img_cv, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'],
                          thickness=2)
            cv2.putText(img_cv, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=last_found['color'], thickness=2)

        cv2.imwrite('./attendImg/'+'check '+img, img_cv)

