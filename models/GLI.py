import numpy as np
import os
from tqdm import tqdm
import itertools
import time
import torch
import torch.nn as nn

class Topology(nn.Module):
    def __init__(self, kinematic_chain, nb_joints):
        super().__init__()
        self.kinematic_chain = kinematic_chain
        self.nb_joints = nb_joints

    def gauss_integral(self, s1, e1, s2, e2):
        """
        Optimized calculation of the Gauss Link Integral (GLI) between two particle chains.

        Parameters:
            s1, e1, s2, e2: torch.tensor (,3)
                3D position vectors representing the start and end points of two chains.

        Returns:
            float: The computed GLI value.
        """
        # Compute vectors between points
        r13, r14 = s2 - s1, e2 - s1
        r23, r24 = s2 - e1, e2 - e1
        r12, r34 = e1 - s1, e2 - s2

        # Calculate face normals and their norms
        faces = [torch.cross(r13, r14), torch.cross(r14, r24), torch.cross(r24, r23), torch.cross(r23, r13)]
        normalized_faces = [vec / torch.norm(vec) if torch.norm(vec) != 0 else torch.zeros(3) for vec in faces]

        # Compute GLI using arcsin of dot products
        GLI = 0.0
        for i in range(4):
            dot = torch.dot(normalized_faces[i], normalized_faces[(i + 1) % 4])
            dot = torch.clip(dot, -1.0, 1.0)  # Clip to handle numerical issues
            GLI += torch.arcsin(dot)

        # Determine the sign of GLI using cross-product and dot-product
        sign = torch.dot(torch.cross(r34, r12), r13)
        GLI *= -1 if sign <= 0 else 1

        # Scale by the normalization factor
        return GLI / (4.0 * torch.pi)


    def gauss_integral_all(self, path1, path2):
        """
        计算两个粒子路径之间的高斯链积分的总和。

        参数:
            path1: list of torch.tensor, 粒子路径 1，每个元素是一个 [x, y, z] 的 tensor 张量。
            path2: list of torch.tensor, 粒子路径 2，每个元素是一个 [x, y, z] 的 tensor 张量。
            gauss_integral_func: function, 用于计算两条链段之间的 GLI 的函数。
        返回:
            float: 高斯链积分总和。
        """

        GLI = 0.0

        # 遍历 path1 的每对连续粒子
        for i in range(len(path1) - 1):
            s1, e1 = path1[i], path1[i + 1]

            # 遍历 path2 的每对连续粒子
            for j in range(len(path2) - 1):
                s2, e2 = path2[j], path2[j + 1]

                # 调用 gauss_integral_func 计算当前两条链段的 GLI
                GLI += self.gauss_integral(s1, e1, s2, e2)

        return GLI

    def gauss_integral_skeleton(self, paths, pose1, pose2):

        GLI = torch.zeros([len(paths), len(paths)])
        for i in range(len(paths)):
            path1 = pose1[paths[i]]
            for j in range(len(paths)):
                path2 = pose2[paths[j]]
                GLI[i, j] = self.gauss_integral_all(path1, path2)
        return GLI

    def gauss_integral_motion(self, motion1, motion2):
        """
        Optimized computation of Gauss Link Integral motion for a sequence of poses.

        Parameters:
            kinematic_chain: list of lists, describing the hierarchy of chains in the skeleton.
            motion1, motion2: torch.tensor, shape (frame_num, joint_num, 3),
                              describing the positions of joints across frames for two motions.

        Returns:
            torch.tensor: Shape (frame_num-1,), containing the maximum absolute velocity of GLI changes across frames.
        """

        paths = [chain[1:] for chain in self.kinematic_chain]
        paths_extra = [[2,14], [1,13]]
        # paths = paths + paths_extra


        frame = min(len(motion1), len(motion2))
        GLI_motion = torch.zeros([frame, len(paths), len(paths)])
        overlap_flags = torch.zeros(frame)
        for i in range(frame):
            bbox1 = self.calculate_xz_bounding_box(motion1[i])
            bbox2 = self.calculate_xz_bounding_box(motion2[i])
            overlap_flags[i] = int(self.check_bounding_box_overlap(bbox1, bbox2))

        windows = []
        count = 0
        while count < frame:
            if overlap_flags[count] == 1:
                # 找到当前窗口的起点和终点，向前后扩展
                start = max(count - 1, 0)
                while count < frame and overlap_flags[count] == 1:
                    count += 1
                end = min(count + 1 - 1, frame - 1)

                # 添加窗口范围
                windows.append((start, end))
            else:
                count += 1

        # 遍历所有帧，计算是否重合
        for start, end in windows:
            for i in range(start, end + 1):
                pose1 = motion1[i]
                pose2 = motion2[i]
                GLI_pose = self.gauss_integral_skeleton(paths, pose1, pose2)
                GLI_motion[i] = GLI_pose

        GLI_abs_vel = torch.abs(GLI_motion[1:] - GLI_motion[:-1])
        GLI_abs_vel = GLI_abs_vel.reshape(frame-1, len(paths) * len(paths))
        GLI_abs_vel_max = torch.max(GLI_abs_vel, dim=1)[0]
        return GLI_abs_vel_max

    def calculate_xz_bounding_box(self, pose):
        x_coords = pose[:, 0]
        z_coords = pose[:, 2]
        min_x, max_x = torch.min(x_coords), torch.max(x_coords)
        min_z, max_z = torch.min(z_coords), torch.max(z_coords)
        return min_x, max_x, min_z, max_z

    def check_bounding_box_overlap(self, bbox1, bbox2):
        min_x1, max_x1, min_z1, max_z1 = bbox1
        min_x2, max_x2, min_z2, max_z2 = bbox2

        # 检查是否在 x 或 z 方向完全分离
        if max_x1 < min_x2 or max_x2 < min_x1:
            return False  # x 方向分离
        if max_z1 < min_z2 or max_z2 < min_z1:
            return False  # z 方向分离

        return True

    def forward(self, motion1, motion2):
        return self.gauss_integral_motion(motion1, motion2)

class TopologyBatch(nn.Module):
    def __init__(self, kinematic_chain, nb_joints):
        super().__init__()
        self.kinematic_chain = kinematic_chain
        self.nb_joints = nb_joints

    def calculate_xz_bounding_box(self, pose):
        # Compute bounding box along batch and frame dimensions
        x_coords, z_coords = pose[..., 0], pose[..., 2]
        return torch.min(x_coords, dim=-1).values, torch.max(x_coords, dim=-1).values, torch.min(z_coords,
                                                                                                 dim=-1).values, torch.max(
            z_coords, dim=-1).values

    def check_bounding_box_overlap(self, bbox1, bbox2):
        # Check for overlap in batch across x and z dimensions
        min_x1, max_x1, min_z1, max_z1 = bbox1
        min_x2, max_x2, min_z2, max_z2 = bbox2
        overlap_x = (max_x1 >= min_x2) & (max_x2 >= min_x1)
        overlap_z = (max_z1 >= min_z2) & (max_z2 >= min_z1)
        return overlap_x & overlap_z

    def gauss_integral(self, s1, e1, s2, e2):
        """
        Optimized calculation of the Gauss Link Integral (GLI) between two particle chains.

        Parameters:
            s1, e1, s2, e2: torch.tensor (,3)
                3D position vectors representing the start and end points of two chains.

        Returns:
            float: The computed GLI value.
        """
        # Compute vectors between points
        r13, r14 = s2 - s1, e2 - s1
        r23, r24 = s2 - e1, e2 - e1
        r12, r34 = e1 - s1, e2 - s2

        # Calculate face normals and their norms
        faces = [torch.cross(r13, r14), torch.cross(r14, r24), torch.cross(r24, r23), torch.cross(r23, r13)]
        normalized_faces = [vec / torch.norm(vec) if torch.norm(vec) != 0 else torch.zeros(3) for vec in faces]

        # Compute GLI using arcsin of dot products
        GLI = 0.0
        for i in range(4):
            dot = torch.dot(normalized_faces[i], normalized_faces[(i + 1) % 4])
            dot = torch.clip(dot, -1.0, 1.0)  # Clip to handle numerical issues
            GLI += torch.arcsin(dot)

        # Determine the sign of GLI using cross-product and dot-product
        sign = torch.dot(torch.cross(r34, r12), r13)
        GLI *= -1 if sign <= 0 else 1

        # Scale by the normalization factor
        return GLI / (4.0 * torch.pi)


    def gauss_integral_all(self, path1, path2):
        """
        计算两个粒子路径之间的高斯链积分的总和。

        参数:
            path1: list of torch.tensor, 粒子路径 1，每个元素是一个 [x, y, z] 的 tensor 张量。
            path2: list of torch.tensor, 粒子路径 2，每个元素是一个 [x, y, z] 的 tensor 张量。
            gauss_integral_func: function, 用于计算两条链段之间的 GLI 的函数。
        返回:
            float: 高斯链积分总和。
        """

        GLI = 0.0

        # 遍历 path1 的每对连续粒子
        for i in range(len(path1) - 1):
            s1, e1 = path1[i], path1[i + 1]

            # 遍历 path2 的每对连续粒子
            for j in range(len(path2) - 1):
                s2, e2 = path2[j], path2[j + 1]

                gli_line = self.gauss_integral(s1, e1, s2, e2)
                GLI += gli_line
                # print('gli_line', gli_line.grad_fn)
        return GLI

    def gauss_integral_skeleton(self, paths, pose1, pose2):

        GLI = torch.zeros([len(paths), len(paths)])
        for i in range(len(paths)):
            path1 = pose1[paths[i]]
            for j in range(len(paths)):
                path2 = pose2[paths[j]]
                gli_bone = self.gauss_integral_all(path1, path2)
                GLI[i, j] = gli_bone
        return GLI

    def gauss_integral_motion(self, motion1, motion2):
        """
        Optimized computation of Gauss Link Integral motion for a sequence of poses.

        Parameters:
            kinematic_chain: list of lists, describing the hierarchy of chains in the skeleton.
            motion1, motion2: torch.tensor, shape (frame_num, joint_num, 3),
                              describing the positions of joints across frames for two motions.

        Returns:
            torch.tensor: Shape (frame_num-1,), containing the maximum absolute velocity of GLI changes across frames.
        """

        paths = [chain[1:] for chain in self.kinematic_chain]
        paths_extra = [[2,14], [1,13]]
        # paths = paths + paths_extra


        batch, frame = motion1.shape[0], motion1.shape[1]
        GLI_motion = torch.zeros([batch, frame, len(paths), len(paths)])
        overlap_flags = torch.zeros([batch, frame])

        for i in range(frame):
            bbox1 = self.calculate_xz_bounding_box(motion1[:, i])
            bbox2 = self.calculate_xz_bounding_box(motion2[:, i])
            overlap_flags[:, i] = self.check_bounding_box_overlap(bbox1, bbox2).int()

        windows = []

        for b in range(batch):
            count = 0
            while count < frame:
                if overlap_flags[b, count]:
                    start = max(count - 1, 0)
                    while count < frame and overlap_flags[b, count]:
                        count += 1
                    end = min(count, frame - 1)
                    windows.append((b, start, end))
                else:
                    count += 1

        for b, start, end in windows:
            for i in range(start, end + 1):
                pose1 = motion1[b, i]
                pose2 = motion2[b, i]
                GLI_pose = self.gauss_integral_skeleton(paths, pose1, pose2)
                GLI_motion[b, i] = GLI_pose

        GLI_abs_vel = torch.abs(GLI_motion[:, 1:] - GLI_motion[:, :-1])
        GLI_abs_vel = GLI_abs_vel.view(batch, frame - 1, -1)
        GLI_abs_vel_max = torch.max(GLI_abs_vel, dim=-1)[0]
        return GLI_abs_vel_max

    def forward(self, motion1, motion2):
        return self.gauss_integral_motion(motion1, motion2)


if __name__ == '__main__':
    motion1 = np.load('../results/multi/left_punch_control_3p_th15_GLI_p0.npy')
    motion2 = np.load('../results/multi/left_punch_control_3p_th15_GLI_p1.npy')
    motion3 = np.load('../results/multi/left_punch_control_3p_th15_GLI_p2.npy')
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    print(motion3)
    motion3 -= 1
    GLI_layer = TopologyBatch(kinematic_chain, 22)
    # GLI_new = gauss_integral_motion(kinematic_chain, motion1, motion2)
    motion1 = torch.from_numpy(np.array([motion1, motion1]))
    motion3 = torch.from_numpy(np.array([motion2, motion3]))

    with torch.enable_grad():
        motion1 = motion1.detach().requires_grad_()
        GLI = GLI_layer(motion1, motion3)
        distance = torch.norm((motion1 - motion3), dim=-1)
        print(GLI.sum())
        print(distance)