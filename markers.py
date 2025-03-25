import mujoco
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation


@dataclass
class MarkerData:
    name: str
    type: mujoco.mjtGeom
    size: np.ndarray
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.3]))

    @property
    def rot_matrix_flat(self) -> np.ndarray:
        match self.rot.ndim:
            case 1:
                if len(self.rot) != 4:
                    raise ValueError(
                        "invalid length of 1D marker rotation: "
                        f"expected scalar-first quaternion with shape (4, ), got {len(self.rot)} "
                    )
                return Rotation.from_quat(self.rot, scalar_first=True).as_matrix().ravel()
            case 2:
                return self.rot.ravel()

            case _:
                raise ValueError(f"wrong ndim of the self.rot, expected 1 <= self.rot.ndim <= 2, got {self.rot.ndim}")

        return np.array(0)

    @property
    def rot_matrix(self) -> np.ndarray:
        match self.rot.ndim:
            case 1:
                if len(self.rot) != 4:
                    raise ValueError(
                        "invalid length of 1D marker rotation: "
                        f"expected scalar-first quaternion with shape (4, ), got {len(self.rot)} "
                    )
                return Rotation.from_quat(self.rot, scalar_first=True).as_matrix()
            case 2:
                return self.rot

            case _:
                raise ValueError(f"wrong ndim of the self.rot, expected 1 <= self.rot.ndim <= 2, got {self.rot.ndim}")

        return np.array(0)


class AxesMarker:
    markers: list[MarkerData]

    def __init__(self, base_pos, size_main=0.05, size_add=0.01, colored=True):
        
        self.markers = [
            MarkerData(
                "a",
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([size_main, size_add, size_add]),  # size
                base_pos + np.array([size_main, 0, 0]),  # pos
                rgba=np.array([1, 0, 0, 0.5]) if colored else np.array([0.5, 0.5, 0.5, 0.5])
            ),
            MarkerData(
                "b",
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([size_add, size_main, size_add]),  # size
                base_pos + np.array([0, size_main, 0]),  # pos
                rgba=np.array([0, 1, 0, 0.5]) if colored else np.array([0.5, 0.5, 0.5, 0.5])
            ),
            MarkerData(
                "c",
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([size_add, size_add, size_main]),  # size
                base_pos + np.array([0, 0, size_main]),  # pos
                rgba=np.array([0, 0, 1, 0.5]) if colored else np.array([0.5, 0.5, 0.5, 0.5])
            ),
        ]
        self.base_pos = base_pos
        self.size_main = size_main

    def update(self, rot):
        for marker in self.markers:
            marker.rot = rot
        self.markers[0].pos = self.base_pos + self.markers[0].rot_matrix @ np.array([self.size_main, 0, 0])
        self.markers[1].pos = self.base_pos + self.markers[1].rot_matrix @ np.array([0, self.size_main, 0])
        self.markers[2].pos = self.base_pos + self.markers[2].rot_matrix @ np.array([0, 0, self.size_main])


class MarkerDrawer:
    def __init__(self, viewer):
        self.markers = []
        self.default_n_geom = viewer.user_scn.ngeom
        self.viewer = viewer

    def draw_markers(self):
        self.viewer.user_scn.ngeom = self.default_n_geom + len(self.markers)
        for i, marker in enumerate(self.markers):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.default_n_geom + i],
                marker.type,
                marker.size,
                marker.pos,
                marker.rot_matrix_flat,
                marker.rgba,
            )
        
    def mark_path(self, path, eef_axis, size=0.03):
        path_marks = [
            MarkerData(
                "",
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([size, 0, 0]),  # size
                Rotation.from_quat(quat, scalar_first=True).as_matrix() @ eef_axis,  # pos
                rgba=np.array([0.1, 0.1, 0.8, 0.4])
            )
            for quat in path
        ]
        self.markers.extend(path_marks)
        return path_marks
    
    def color_path(self, path_marks, idx):
        for mark in path_marks:
            mark.rgba = np.array([0.1, 0.1, 0.8, 0.4])
        if idx >= len(path_marks):
            return
        else:
            path_marks[idx].rgba = np.array([0.1, 0.8, 0.1, 0.4])