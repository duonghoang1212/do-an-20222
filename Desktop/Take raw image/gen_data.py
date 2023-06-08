from PIL import Image, ImageTk

import cv2
import os
import csv
import numpy as np
import json
import shutil

import cv2
import numpy as np
import math

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)

def generate_dataset(self):
# thêm vào đây
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    cap = cv2.VideoCapture(0)  # batcam
    ret, img = cap.read()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
    )

    img_id1 = 0
    img_id2 = 0
    img_id3 = 0
    img_id4 = 0
    img_id5 = 0
    img_id6 = 0
    while True:
        ret, img = cap.read()

        if ret == True:
            faces = find_faces(img, face_model)

            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                    marks[30],  # Nose tip
                    marks[8],  # Chin
                    marks[36],  # Left eye left corner
                    marks[45],  # Right eye right corne
                    marks[48],  # Left Mouth corner
                    marks[54]  # Right mouth corner
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                            camera_matrix,
                                                                            dist_coeffs,
                                                                            flags=cv2.SOLVEPNP_UPNP)

                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                                                rotation_vector,
                                                                translation_vector, camera_matrix,
                                                                dist_coeffs)
                # for p in image_points:
                #     cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                # cv2.line(img, p1, p2, (0, 255, 255), 2)
                # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                # cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1 / m)))
                except:
                    ang2 = 90

                    # print('div by zero error')

                if img_id6 < 20:
                    img_id1 = img_id1 + 1
                    img_id6 = img_id6 + 1
                    face1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/images/" + str(self) + "/" + str(img_id1) + ".jpg"
                    cv2.imwrite(file_name_path, face1)
                    cv2.putText(face1, str(img_id1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("cropped face", face1)

                if img_id4 < 10:
                    print('nên Head left')
                    cv2.putText(img, 'NEN QUAY TRAI', (90, 30), font, 2, (255, 255, 128), 3)

                if img_id3 < 10 and img_id4 == 10:
                    print('NEN Head right')
                    cv2.putText(img, 'NEN QUAY PHAI', (90, 30), font, 2, (255, 255, 128), 3)

                if img_id2 < 5 and img_id3 == 10 and img_id4 == 10:
                    print('NEN Head up')
                    cv2.putText(img, 'NEN QUAY LEN', (30, 30), font, 2, (255, 255, 128), 3)

                if img_id5 < 5 and img_id2 == 5 and img_id3 == 10 and img_id4 == 10:
                    print('NEN Head down')
                    cv2.putText(img, 'NEN QUAY XUONG', (30, 30), font, 2, (255, 255, 128), 3)

                if ang1 >= 48 and img_id5 < 5:

                    print('Head down')
                    cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                    img_id1 = img_id1 + 1
                    img_id5 = img_id5 + 1
                    face1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/images/" + str(self) + "/" + str(img_id1) + ".jpg"
                    cv2.imwrite(file_name_path, face1)
                    cv2.putText(face1, str(img_id1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("cropped face", face1)



                elif ang1 <= -48 and img_id2 < 5:
                    print('Head up')
                    cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
                    img_id1 = img_id1 + 1
                    img_id2 = img_id2 + 1
                    face1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/images/" + str(self) + "/" + str(img_id1) + ".jpg"
                    cv2.imwrite(file_name_path, face1)
                    cv2.putText(face1, str(img_id1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("cropped face", face1)

                if ang2 >= 48 and img_id3 < 10:
                    print('Head right')
                    cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                    img_id1 = img_id1 + 1
                    img_id3 = img_id3 + 1
                    face1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/images/" + str(self) + "/" + str(img_id1) + ".jpg"
                    cv2.imwrite(file_name_path, face1)
                    cv2.putText(face1, str(img_id1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("cropped face", face1)




                elif ang2 <= -48 and img_id4 < 10:
                    print('Head left')
                    cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                    img_id1 = img_id1 + 1
                    img_id4 = img_id4 + 1
                    face1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/images/" + str(self) + "/" + str(img_id1) + ".jpg"
                    cv2.imwrite(file_name_path, face1)
                    cv2.putText(face1, str(img_id1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("cropped face", face1)

                # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q') or int(img_id1) == 50:
                cv2.destroyAllWindows()
                cap.release()
                break
        else:
            break
    ## cv2.destroyAllWindows()
    ## cap.release()

    print("oke")
# except Exception as es:
#     messagebox.showerror("Error", f"due to :{str(es)}", parent=self.root)

# ======================================================================== sau khi đã lấy ảnh thành công, chạy lệnh để gọi chương trình trainning của my_ai
