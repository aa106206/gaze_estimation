"""GazeHub eye-based 정규화 코어.

GazeHub `data_processing_core.py`를 이 레포용으로 옮긴 것이다.
수학은 Zhang et al. / Sugano et al.의 perspective normalization과 같다.

- 새 초점거리 `newfocal=960`, 정규화 거리 `newdistance=600`
- `imsize=(width, height)` — MPIIGaze 눈 이미지는 (60, 36)
"""

import numpy as np
import cv2


class Norm:
    def __init__(
        self,
        center,
        gazetarget,
        headrotvec,
        imsize,
        camparams,
        newfocal=960,
        newdistance=600,
    ):
        self.center = np.array(center, dtype=np.float64)
        self.headrotvec = np.array(headrotvec, dtype=np.float64)
        self.target = np.array(gazetarget, dtype=np.float64)
        self.imsize = np.array(imsize, dtype=np.float64)
        self.cameraparams = np.array(camparams, dtype=np.float64)
        self.newfocal = newfocal
        self.newdistance = newdistance
        self.curdistance = np.linalg.norm(self.center)

        self._assert_inputs()
        if self.headrotvec.shape == (3,):
            self.headrotvec = cv2.Rodrigues(self.headrotvec)[0]
        self._params()

    def _assert_inputs(self):
        assert self.center.shape == (3,)
        assert self.headrotvec.shape in ((3,), (3, 3))
        assert self.target.shape == (3,)
        assert self.imsize.shape == (2,)
        assert self.cameraparams.shape == (3, 3)

    def _params(self):
        self.S_mat = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, self.newdistance / self.curdistance]]
        )
        xaxis = self.headrotvec[:, 0]
        z = self.center / self.curdistance
        y = np.cross(z, xaxis)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        self.R_mat = np.array([x, y, z])
        self.C_mat = np.array(
            [
                [self.newfocal, 0.0, self.imsize[0] / 2.0],
                [0.0, self.newfocal, self.imsize[1] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.M_mat = np.dot(self.S_mat, self.R_mat)
        self.W_mat = np.dot(np.dot(self.C_mat, self.M_mat), np.linalg.inv(self.cameraparams))
        self.gaze = self.target - self.center

    def GetParams(self):
        rvec = cv2.Rodrigues(self.R_mat)[0].flatten()
        svec = np.diagonal(self.S_mat)
        return rvec, svec

    def GetImage(self, image):
        return cv2.warpPerspective(
            image, self.W_mat, (int(self.imsize[0]), int(self.imsize[1]))
        )

    def GetCoordinate(self, coordinate):
        coordinate = np.reshape(coordinate, (3, 1))
        return np.dot(self.M_mat, coordinate).flatten()

    def GetGaze(self, scale=True):
        mat = self.M_mat if scale else self.R_mat
        gaze = np.dot(mat, self.gaze)
        return gaze / np.linalg.norm(gaze)

    def GetHeadRot(self, vector=True):
        rotated = np.dot(self.M_mat, self.headrotvec)
        if vector:
            return cv2.Rodrigues(rotated)[0].flatten()
        return rotated


def gaze_to_2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


def gaze_to_3d(gaze):
    """`gaze_to_2d`의 역변환. (yaw, pitch) → 3D 단위 벡터. (..., 2) 배열도 받는다."""
    gaze = np.asarray(gaze, dtype=np.float64)
    yaw, pitch = gaze[..., 0], gaze[..., 1]
    return np.stack(
        [-np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)], axis=-1
    )


def head_to_2d(head):
    head = np.asarray(head)
    assert head.shape == (3,), f"head rotvec must be (3,), got {head.shape}"
    M = cv2.Rodrigues(head)[0]
    vec = M[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([yaw, pitch])


def gaze_flip(gaze):
    newgaze = np.zeros(3)
    newgaze[0] = -gaze[0]
    newgaze[1] = gaze[1]
    newgaze[2] = gaze[2]
    return newgaze


def head_flip(head):
    """GazeHub `HeadFlip`과 동일. rot_mat의 z축 x성분을 in-place로 뒤집은 뒤 Rodrigues를 다시 푼다."""
    rot_vec = np.array(head)
    assert rot_vec.shape == (3,), f"head rotvec must be (3,), got {rot_vec.shape}"
    rot_mat = cv2.Rodrigues(rot_vec)[0]
    z = rot_mat[:, 2]
    z[0] = -z[0]
    return cv2.Rodrigues(rot_mat)[0].flatten()
