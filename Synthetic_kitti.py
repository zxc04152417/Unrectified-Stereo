# -*- coding: utf-8 -*-
import os
import re
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R
from glob import glob

# ============================= I/O Utils =================================

def parse_calib_file12(calib_path):
    """解析KITTI标定文件，返回P0-P3投影矩阵"""
    import numpy as np

    calib_data = {}

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P'):
                # 分割行，获取标签和数据
                parts = line.split()
                key = parts[0].rstrip(':').lower()  # 去掉冒号并转为小写
                values = [float(x) for x in parts[1:]]  # 转换为浮点数

                # 重新组织为3x4矩阵
                calib_data[key] = np.array(values).reshape(3, 4)

    return calib_data

def parse_calib_cam_to_cam15(calib_path):
    """读取 KITTI2015 相机内参 K_00~K_03"""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('K_'):
                parts = line.split()
                key = parts[0].rstrip(':')
                vals = [float(x) for x in parts[1:]]
                calib[key] = np.array(vals).reshape(3, 3)
    return calib

def get_intrinsics_from_path(file_path):
    """按路径推内参（示例逻辑：15mm / 35mm 两档）。"""
    path_parts = file_path.split(os.sep)
    a_param = path_parts[-4]
    if a_param in ['scene_backwards_15', 'scene_forwards_15']:
        intrinsics = [
            [450.0, 0.0, 479.5],
            [0.0, 450.0, 269.5],
            [0.0, 0.0,   1.0]
        ]
    else:
        intrinsics = [
            [1050.0, 0.0, 479.5],
            [0.0, 1050.0, 269.5],
            [0.0, 0.0,    1.0]
        ]
    return np.array(intrinsics, dtype=np.float64)  # 用 float64 提高数稳

def create_custom_folder(file_path: str, suffix: str):
    parent_dir = os.path.dirname(file_path)
    grandparent_dir = os.path.dirname(parent_dir)
    last_dir_name = os.path.basename(parent_dir)
    new_dir_name = last_dir_name + "_" + suffix
    new_folder_path = os.path.join(grandparent_dir, new_dir_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"文件夹已创建: {new_folder_path}")
    else:
        print(f"文件夹已存在: {new_folder_path}")
    return new_folder_path

def transform_path(file_path, old_dir='frames_finalpass', new_dir='camera_intrinc',
                   new_filename='camera_data.txt', remove_last_dir='left'):
    new_path = file_path.replace(old_dir, new_dir)
    dir_path = os.path.dirname(new_path)
    parent_dir = os.path.dirname(dir_path)
    last_dir = os.path.basename(dir_path)
    if last_dir == remove_last_dir:
        dir_path = parent_dir
    final_path = os.path.join(dir_path, new_filename)
    return final_path

def extract_last_number(file_path):
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', name_without_ext)
    return int(numbers[-1]) if numbers else None

def save_matrices(H_total_center: np.ndarray, H_inv_center: np.ndarray, txt_path: str):
    """落盘保存【中心系】的 H_total/H_inv（含扩展与裁剪平移）。"""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write("H_total: " + " ".join(f"{v:.8e}" for v in H_total_center.flatten()) + "\n")
        f.write("H_inv: "   + " ".join(f"{v:.8e}" for v in H_inv_center.flatten())   + "\n")
    print(f"[save] wrote homographies (CENTER coords) -> {txt_path}")

def load_matrices(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    H_total = H_inv = None
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('H_total:'):
                vals = [float(x) for x in line.replace('H_total:', '').strip().split()]
                H_total = np.array(vals).reshape(3, 3)
            elif line.startswith('H_inv:'):
                vals = [float(x) for x in line.replace('H_inv:', '').strip().split()]
                H_inv = np.array(vals).reshape(3, 3)
    if H_total is None or H_inv is None:
        raise ValueError("Bad homography file")
    return H_total, H_inv

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    return disp, (disp > 0.0)

def saveDispKITTI(disp, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, (disp * 256.0).astype(np.float32))

def readPFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    endian = '<' if scale < 0 else '>'
    if scale < 0: scale = -scale
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    assert type(file) is str and type(array) is np.ndarray and os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())

# ======================== Geometry Helpers (core) =========================

def T_shift(dx: float, dy: float) -> np.ndarray:
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = dx
    T[1, 2] = dy
    return T

def to_center(H_corner: np.ndarray) -> np.ndarray:
    """角点系 -> 像素中心系"""
    return T_shift(+0.5, +0.5) @ H_corner @ T_shift(-0.5, -0.5)

def to_corner(H_center: np.ndarray) -> np.ndarray:
    """像素中心系 -> 角点系（仅在 cv2.warpPerspective 前使用）"""
    return T_shift(-0.5, -0.5) @ H_center @ T_shift(+0.5, +0.5)

def apply_homography_centered(H_center: np.ndarray, x: np.ndarray, y: np.ndarray, eps: float = 1e-12):
    """
    在像素中心坐标系下应用单应：输入/输出都以像素中心为基准。
    注意：x,y 可以是一维/网格；返回与 x,y 同形。
    """
    ones = np.ones_like(x)
    xy1 = np.stack([x + 0.5, y + 0.5, ones], axis=0)   # 进中心系
    xyz = H_center @ xy1.reshape(3, -1)                # 3 x (HW)
    z = xyz[2, :].reshape(x.shape)
    valid = np.abs(z) > eps
    # 为避免除 0：对无效位置用极小量（带符号）稳定除法
    den = z + (~valid) * (np.sign(z) * eps) + (~valid) * eps
    xh = (xyz[0, :].reshape(x.shape) / den) - 0.5
    yh = (xyz[1, :].reshape(y.shape) / den) - 0.5
    return xh, yh, valid

def compute_expanded_canvas_and_crop(H_center: np.ndarray, W: int, Ht: int):
    """
    在【中心系】里计算最小外接画布+平移，并合成“直接输出到裁剪坐标”的 H_out_center。
    返回：
      H_out_center:  直接 warp 到裁剪后画布（仍是中心系）
      H_big_center:  仅加了平移到大画布的中间结果（中心系）
      crop_xywh:     (x0, y0, w, h) 在大画布中的裁剪框
      big_size:      (W_big, H_big)
    """
    # 取四角（角点系整数）并按中心系投影
    corners = np.array([[0,    0   ],
                        [W-1., 0   ],
                        [W-1., Ht-1.],
                        [0,    Ht-1.]], dtype=np.float64)
    xs = corners[:, 0]
    ys = corners[:, 1]

    # 角点 -> 中心 -> 变换 -> 角点（apply_homography_centered 已经做了 +0.5/-0.5 ）
    xh, yh, _ = apply_homography_centered(H_center, xs, ys)
    minx, miny = np.min(xh), np.min(yh)
    maxx, maxy = np.max(xh), np.max(yh)

    # 对大角度给足安全边距
    pad = 16.0
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    # 平移使边界 >= 0（仍在中心系）
    tx = -min(0.0, minx)
    ty = -min(0.0, miny)
    H_big_center = T_shift(tx, ty) @ H_center

    # 扩展画布大小（向上取整）
    W_big = int(math.ceil(max(maxx + tx + 1, W)))
    H_big_img = int(math.ceil(max(maxy + ty + 1, Ht)))

    # 尽量居中裁回原尺寸（若扩展不比原大，退化为原尺寸）
    x0 = max(0, (W_big - W) // 2)
    y0 = max(0, (H_big_img - Ht) // 2)
    crop_xywh = (x0, y0, W, Ht)

    # 直接输出到“裁剪后画布”的单应（中心系）
    H_out_center = T_shift(-x0, -y0) @ H_big_center
    return H_out_center, H_big_center, crop_xywh, (W_big, H_big_img)

# ============================== Simulator =================================

class StereoSimulatorV2:
    """
    - 内部所有几何用【像素中心系】单应；
    - 只有在 warpPerspective 前，临时 to_corner() 一次；
    - 落盘的 H_total/H_inv 就是中心系版本。
    """

    def __init__(self,
                 left_img: np.ndarray,
                 right_img: np.ndarray,
                 rect_disp_occ: np.ndarray,
                 K_right: np.ndarray,
                 baseline: float,
                 test_large_offset: bool = False):
        self.left_image  = left_img
        self.right_image = right_img
        self.rectified_disparity_occ = rect_disp_occ.astype(np.float64)
        self.K = K_right.astype(np.float64)
        self.baseline = float(baseline)

        self.H_img, self.W_img = left_img.shape[:2]
        self.setup_realistic_parameters(test_large_offset)
        self.build_homography()

    def setup_realistic_parameters(self, large=False):
        """采样相机姿态/竖向像素漂移（单位：度、像素）。"""
        if large:
            # 你给出的“大量级”设定
            self.rotation_angles = {
                'roll':  np.random.uniform(-1.2,  1.2),
                'pitch': np.random.uniform(-1.0,  1.0),
                'yaw':   np.random.uniform(-1.5,  1.5),
            }
            self.vertical_offset = np.random.uniform(-4.0, 4.0)



        else:
            self.rotation_angles = {
                'roll':  np.random.uniform(-0.5,  0.5),
                'pitch': np.random.uniform(-0.3,  0.3),
                'yaw':   np.random.uniform(-0.8,  0.8),
            }
            self.vertical_offset = np.random.uniform(-0.5, 0.5)

        print(f"[sim] roll={self.rotation_angles['roll']:.3f}°, "
              f"pitch={self.rotation_angles['pitch']:.3f}°, "
              f"yaw={self.rotation_angles['yaw']:.3f}°, "
              f"dy={self.vertical_offset:.3f} px")

    def build_homography(self):
        """构建中心系单应 & 自动扩展画布并裁回原尺寸。"""
        # 角点系下的像素坐标单应（仅姿态）
        Rm = R.from_euler(
            'zyx',
            [self.rotation_angles['yaw'],
             self.rotation_angles['pitch'],
             self.rotation_angles['roll']],
            degrees=True
        ).as_matrix()
        K = self.K
        Kinv = np.linalg.inv(K)
        H_rot_corner = K @ Rm @ Kinv

        # 角点系下的像素平移（只做屏幕像素平移模拟）
        H_trans_corner = np.eye(3, dtype=np.float64)
        H_trans_corner[0, 2] = 0.0
        H_trans_corner[1, 2] = self.vertical_offset

        # 合成后转入中心系做后续计算
        H_rc_corner = H_trans_corner @ H_rot_corner
        H_rc_center = to_center(H_rc_corner)

        # 扩展画布+回裁（中心系）
        H_out_c, H_big_c, crop_xywh, big_size = compute_expanded_canvas_and_crop(
            H_rc_center, self.W_img, self.H_img
        )
        self.H_total_center = H_out_c.copy()
        self.H_inv_center   = np.linalg.inv(self.H_total_center)
        self.crop_xywh      = crop_xywh
        self.big_size       = big_size

        # 条件数：数值稳定性参考
        cond = np.linalg.cond(self.H_total_center)
        print(f"[H] cond(H_total_center) = {cond:.2e} | big_canvas={big_size} | crop={crop_xywh}")

    def warp_right_to_original_size(self) -> np.ndarray:
        """右图 warp：中心系 -> (仅在此处)转角点系 -> OpenCV warp。"""
        H_corner = to_corner(self.H_total_center)  # 仅此处需要角点系
        out = cv2.warpPerspective(
            self.right_image, H_corner,
            (self.W_img, self.H_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )
        return out

    # ---------------------- disparity field (vectorized) -------------------

    def _disparity_field_core(self, rectified_disp) -> Tuple[np.ndarray, np.ndarray]:
        """
        给定 rectified 的水平视差 d_r，计算原始域 (h,v)（中心系单应）。
        输出 (h,v) 用角点索引网格存储。
        """
        H, W = self.H_img, self.W_img
        y, x = np.mgrid[0:H, 0:W].astype(np.float64)

        # rectified 右图坐标（角点系）：(x - d_r, y)
        xr = x - rectified_disp
        yr = y

        # 投到（裁剪后的）原始右图坐标（中心系单应）
        xw, yw, valid = apply_homography_centered(self.H_total_center, xr, yr)

        # 放宽边界容差，避免因半像素/插值导致的边缘丢失
        epsb = 1e-6
        in_img = (xw >= -epsb) & (xw <= W - 1 + epsb) & (yw >= -epsb) & (yw <= H - 1 + epsb) & valid

        h = np.zeros((H, W), dtype=np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        # h[in_img] = x[in_img] - xw[in_img]
        # v[in_img] = y[in_img] - yw[in_img]
        h =(x-xw)*valid
        v=(y-yw)*valid

        return h, v

    def compute_disparity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """用 occ 视差生成 (h,v)。"""
        h_occ, v_occ = self._disparity_field_core(self.rectified_disparity_occ)
        return h_occ, v_occ

    # ---------------------- inverse verification ---------------------------

    def inverse_transform_verification(self, h: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        从原始域 (h,v) 逆到 rectified 的水平 d_r（中心系单应）。
        """
        H, W = self.H_img, self.W_img
        y, x = np.mgrid[0:H, 0:W].astype(np.float64)

        xr = x - h
        yr = y - v

        x_rect, y_rect, valid = apply_homography_centered(self.H_inv_center, xr, yr)

        d_r = np.zeros((H, W), dtype=np.float64)
        use = ((np.abs(h) + np.abs(v)) > 0) & valid
        d_r[use] = x[use] - x_rect[use]
        return d_r

    # ---------------------- error & viz ------------------------------------

    def calculate_errors(self, recovered: np.ndarray) -> Dict[str, float]:
        gt = self.rectified_disparity_occ
        mask1 = gt > 0.0001
        mask2 = recovered > 0.00001
        #valid = mask1 & mask2
        valid = mask1
        if valid.sum() == 0:
            return dict(mean_error=0, max_error=0, rmse=0, std_error=0,
                        median_error=0, valid_pixels=0)
        diff = np.abs(recovered[valid] - gt[valid])
        return dict(
            mean_error=float(np.mean(diff)),
            max_error=float(np.max(diff)),
            rmse=float(np.sqrt(np.mean(diff**2))),
            std_error=float(np.std(diff)),
            median_error=float(np.median(diff)),
            valid_pixels=int(valid.sum()),
        )

    def visualize_results(self, left_o, right_o, h, v, recovered):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0,0].imshow(cv2.cvtColor(left_o,  cv2.COLOR_BGR2RGB));  axes[0,0].set_title('左图');  axes[0,0].axis('off')
        axes[0,1].imshow(cv2.cvtColor(right_o, cv2.COLOR_BGR2RGB));  axes[0,1].set_title('右图(变换后)'); axes[0,1].axis('off')
        im2 = axes[0,2].imshow(h, cmap='jet'); plt.colorbar(im2, ax=axes[0,2]); axes[0,2].set_title('水平视差 h'); axes[0,2].axis('off')
        im3 = axes[1,0].imshow(v, cmap='RdBu_r'); plt.colorbar(im3, ax=axes[1,0]); axes[1,0].set_title('垂直视差 v'); axes[1,0].axis('off')
        im4 = axes[1,1].imshow(self.rectified_disparity_occ, cmap='jet'); plt.colorbar(im4, ax=axes[1,1]); axes[1,1].set_title('rectified GT'); axes[1,1].axis('off')
        im5 = axes[1,2].imshow(recovered, cmap='jet'); plt.colorbar(im5, ax=axes[1,2]); axes[1,2].set_title('逆变换恢复 d_r'); axes[1,2].axis('off')
        plt.tight_layout(); plt.show()

# ============================ Demo Pipeline ===============================

def demo_with_synthetic_data_v2(test_large_offset=False):
    # 根路径（按需改）
    # KITTI15 = "KITTI15_roll_0.2"
    # root = f"/media/kemove/zxc/stereo/roation/roll/{KITTI15}"       #现在探究旋转的影响。
    # dir_left = f"{root}/training/image_2"
    # dir_right = f"{root}/training/image_3"
    # dir_right_out = f"{root}/training/image_3_st"  # 右图(合成)输出
    # dir_disp_occ = f"{root}/training/disp_occ_0"
    # dir_disp_noc = f"{root}/training/disp_noc_0"
    # dir_h = f"{root}/training/disp_occ_0_h"
    # dir_v = f"{root}/training/disp_occ_0_v"
    # dir_h_noc = f"{root}/training/disp_noc_0_h"
    # dir_v_noc = f"{root}/training/disp_noc_0_v"
    # dir_calib = f"{root}/data_scene_flow_calib/training/calib_cam_to_cam"
    # dir_out_H = f"{root}/training/calib_invs"

    KITTI12="KITTI12_roll_0"
    root12 = f"/media/kemove/zxc/stereo/roation/max/{KITTI12}"
    dir_left = f"{root12}/training/colored_0"  # 修改为你的实际路径
    dir_right = f"{root12}/training/colored_1"
    dir_disp_occ = f"{root12}/training/disp_occ"
    dir_calib = f"{root12}/training/calib"
    dir_right_out = f"{root12}/training/colored_1_st"
    gt_dir_noc = f"{root12}/training/disp_noc"  # 修改为你的实际路径
    dir_h = f"{root12}/training/disp_occ_h"
    dir_v = f"{root12}/training/disp_occ_v"
    dir_h_noc = f"{root12}/training/disp_noc_h"
    dir_v_noc = f"{root12}/training/disp_noc_v"
    dir_out_H = f"{root12}/training/calib_invs"


    #dir_left  = "/media/kemove/Elements/zxc/sceneflow/frames_finalpass/TEST/*/*/left/*.png
    os.makedirs(dir_right_out, exist_ok=True)
    os.makedirs(dir_h, exist_ok=True);
    os.makedirs(dir_v, exist_ok=True)
    os.makedirs(dir_h_noc, exist_ok=True);
    os.makedirs(dir_v_noc, exist_ok=True)
    os.makedirs(dir_out_H, exist_ok=True)


    names = sorted(os.listdir(dir_disp_occ))
    for name in names:
        pL = os.path.join(dir_left, name)
        pR = os.path.join(dir_right, name)
        if not (os.path.isfile(pL) and os.path.isfile(pR)):  # 跳过非图像
            continue

        left  = cv2.imread(pL)
        right = cv2.imread(pR)

        # calib_path = re.sub(r'_\d+\.png$', '.txt', os.path.join(dir_calib, name))
        # calib = parse_calib_cam_to_cam15(calib_path)
        # K_right = calib['K_03'][:, :3]

        calib_path = re.sub(r'_\d+\.png$', '.txt', os.path.join(dir_calib, name))
        calib_data = parse_calib_file12(calib_path)
        #height, width = left_img.shape[:2]

        K_left = calib_data['p2']
        K_right = calib_data['p3']
        #K_left = K_left[:, :3]
        K_right= K_right[:, :3]

        p_occ = os.path.join(dir_disp_occ, name)
        disp, _ = readDispKITTI(p_occ)



        # 构建模拟器（large=True 使用大角度+大竖移采样）
        sim = StereoSimulatorV2(left, right, disp, K_right, baseline=0.54,
                                test_large_offset=test_large_offset)

        # 右图合成并落盘
        right_out = sim.warp_right_to_original_size()
        cv2.imwrite(os.path.join(dir_right_out, name), right_out)

        # 生成 (h,v) 并落盘（pfm）
        h_occ, v_occ = sim.compute_disparity_field()
        h_occ = np.float32(h_occ)
        v_occ = np.float32(v_occ)

        saveDispKITTI(h_occ, os.path.join(dir_h,     name.replace('.png', '.tiff')))
        saveDispKITTI(v_occ, os.path.join(dir_v,     name.replace('.png', '.tiff')))

        # 保存【中心系】H_total/H_inv（txt）
        # txt_path = os.path.join(H_inv_root, os.path.basename(disp_root)).replace('.pfm', '.txt')
        # save_matrices(sim.H_total_center, sim.H_inv_center, txt_path)
        save_matrices(sim.H_total_center, sim.H_inv_center, os.path.join(dir_out_H, name.replace('.png', '.txt')))


        # 逆验证（由 (h,v) -> rectified d）
        recovered = sim.inverse_transform_verification(h_occ, v_occ)
        errs = sim.calculate_errors(recovered)
        print(f"[verify] {name}: mean={errs['mean_error']:.6f}, median={errs['median_error']:.6f}, "
              f"rmse={errs['rmse']:.6f}, max={errs['max_error']:.6f}, n={errs['valid_pixels']}")

        # 如需可视化，取消注释
        #sim.visualize_results(left, right_out, h_occ, v_occ, recovered)

    return 0

# =============================== Main =====================================

if __name__ == "__main__":
    print("=== V2: normal offset ===")
    # demo_with_synthetic_data_v2(test_large_offset=False)

    print("=== V2: large offset ===")
    demo_with_synthetic_data_v2(test_large_offset=True)