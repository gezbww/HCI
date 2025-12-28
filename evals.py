# import os
# import glob
# import json
# import time
# from typing import List, Tuple, Dict

# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from torchvision.models import inception_v3, resnet18

# import lpips
# import face_alignment
# from scipy import linalg


# # =========================
# # 工具：遍历视频帧
# # =========================

# def read_video_frames(path: str) -> List[Image.Image]:
#     """读取 mp4，返回 PIL Image 列表（RGB）"""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {path}")
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # BGR -> RGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))
#     cap.release()
#     return frames


# def get_video_fps(path: str) -> float:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0.0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return fps


# # =========================
# # FID: Inception 特征
# # =========================

# class InceptionFeatureExtractor(nn.Module):
#     def __init__(self, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.model = inception_v3(pretrained=True, transform_input=False)
#         self.model.fc = nn.Identity()
#         self.model.eval().to(device)

#         self.transform = T.Compose([
#             T.Resize((299, 299)),
#             T.ToTensor(),
#             T.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
#         ])

#     @torch.no_grad()
#     def get_features_from_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
#         feats = []
#         for i in range(0, len(images), batch_size):
#             batch_imgs = images[i:i+batch_size]
#             if not batch_imgs:
#                 continue
#             tensors = [self.transform(im) for im in batch_imgs]
#             x = torch.stack(tensors, dim=0).to(self.device)
#             with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
#                 f = self.model(x)
#             feats.append(f.detach().cpu().numpy())
#         if len(feats) == 0:
#             return np.zeros((0, 2048), dtype=np.float32)
#         feats = np.concatenate(feats, axis=0)
#         return feats


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     diff = mu1 - mu2
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     tr_covmean = np.trace(covmean)
#     fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#     return float(fid)


# def compute_stats(feats: np.ndarray):
#     mu = np.mean(feats, axis=0)
#     sigma = np.cov(feats, rowvar=False)
#     return mu, sigma


# def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
#     mu_r, sig_r = compute_stats(real_feats)
#     mu_f, sig_f = compute_stats(fake_feats)
#     fid = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)
#     return fid


# # =========================
# # LPIPS
# # =========================

# class LPIPSMetric:
#     def __init__(self, device="cuda"):
#         self.device = device
#         self.loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
#         self.transform = T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor()
#         ])
#         self.values = []

#     @torch.no_grad()
#     def update(self, gt_img: Image.Image, pr_img: Image.Image):
#         gt = self.transform(gt_img).unsqueeze(0).to(self.device)
#         pr = self.transform(pr_img).unsqueeze(0).to(self.device)
#         v = self.loss_fn(gt, pr)
#         self.values.append(v.item())

#     def compute(self):
#         return float(np.mean(self.values)) if self.values else None


# # =========================
# # Landmark Error (NME)
# # =========================

# class LandmarkErrorMetric:
#     def __init__(self, device="cuda"):
#         self.device = device

#         # 兼容不同版本的 face_alignment：
#         # 老版本: LandmarksType._2D
#         # 新版本: LandmarksType.TWO_D
#         try:
#             lm_type = face_alignment.LandmarksType._2D
#         except AttributeError:
#             lm_type = face_alignment.LandmarksType.TWO_D

#         self.fa = face_alignment.FaceAlignment(
#             lm_type,
#             flip_input=False,
#             device=device
#         )
#         self.errors = []


#     def _get_landmarks(self, img: Image.Image):
#         arr = np.array(img)
#         preds = self.fa.get_landmarks(arr)
#         if preds is None or len(preds) == 0:
#             return None
#         return preds[0]  # (K,2)

#     def update(self, gt_img: Image.Image, pr_img: Image.Image):
#         gt_lm = self._get_landmarks(gt_img)
#         pr_lm = self._get_landmarks(pr_img)
#         if gt_lm is None or pr_lm is None:
#             return
#         K = min(gt_lm.shape[0], pr_lm.shape[0])
#         gt_lm = gt_lm[:K]
#         pr_lm = pr_lm[:K]

#         # 用两眼中心距离做归一化
#         left_eye = gt_lm[36:42].mean(axis=0)
#         right_eye = gt_lm[42:48].mean(axis=0)
#         d = np.linalg.norm(left_eye - right_eye) + 1e-6

#         err = np.linalg.norm(gt_lm - pr_lm, axis=1).mean() / d
#         self.errors.append(err)

#     def compute(self):
#         return float(np.mean(self.errors)) if self.errors else None


# # =========================
# # Expression Consistency
# # =========================

# class ExpressionConsistencyMetric:
#     """
#     这里用 ResNet18 的特征做一个“表情 embedding”示例。
#     若你有专门的 expression encoder，可以在这里替换。
#     """
#     def __init__(self, device="cuda"):
#         self.device = device
#         backbone = resnet18(pretrained=True)
#         backbone.fc = nn.Identity()
#         self.encoder = backbone.to(device).eval()
#         self.transform = T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
#         self.sims = []

#     @torch.no_grad()
#     def _encode(self, img: Image.Image) -> torch.Tensor:
#         x = self.transform(img).unsqueeze(0).to(self.device)
#         feat = self.encoder(x)
#         feat = torch.nn.functional.normalize(feat, dim=-1)
#         return feat

#     def update(self, gt_img: Image.Image, pr_img: Image.Image):
#         f_gt = self._encode(gt_img)
#         f_pr = self._encode(pr_img)
#         sim = (f_gt * f_pr).sum(dim=-1)  # cosine similarity
#         self.sims.append(sim.item())

#     def compute(self):
#         return float(np.mean(self.sims)) if self.sims else None


# # =========================
# # 对齐视频 & 主流程
# # =========================

# def pair_videos(gt_dir: str, pred_dir: str) -> List[Tuple[str, str]]:
#     """
#     按文件名对齐：pred_dir 中有 foo.mp4，就在 gt_dir 里找同名 foo.mp4。
#     """
#     gt_map = {}
#     for p in glob.glob(os.path.join(gt_dir, "*.mp4")):
#         name = os.path.basename(p)
#         gt_map[name] = p

#     pairs = []
#     for p in glob.glob(os.path.join(pred_dir, "*.mp4")):
#         name = os.path.basename(p)
#         if name in gt_map:
#             pairs.append((gt_map[name], p))
#     return pairs


# def evaluate_all(gt_dir: str, pred_dir: str, device: str = "cuda") -> Dict:
#     pairs = pair_videos(gt_dir, pred_dir)
#     print(f"[INFO] Found {len(pairs)} paired videos.")

#     if len(pairs) == 0:
#         raise RuntimeError("No paired videos found. Make sure filenames in gt_dir and pred_dir match.")

#     # 统计播放 FPS（注意：这不是推理延迟，只是视频元数据）
#     fps_list = []
#     for gt_path, pred_path in pairs:
#         fps = get_video_fps(pred_path)
#         fps_list.append(fps)
#     avg_playback_fps = float(np.mean(fps_list))

#     # 指标对象
#     lpips_metric = LPIPSMetric(device=device)
#     landmark_metric = LandmarkErrorMetric(device=device)
#     expr_metric = ExpressionConsistencyMetric(device=device)
#     inception_extractor = InceptionFeatureExtractor(device=device)

#     real_feats_all = []
#     fake_feats_all = []

#     # 遍历所有视频
#     for gt_path, pred_path in tqdm(pairs, desc="Evaluating videos"):
#         # 解码视频
#         gt_frames = read_video_frames(gt_path)
#         pr_frames = read_video_frames(pred_path)

#         # 对齐帧数（取较短的部分）
#         L = min(len(gt_frames), len(pr_frames))
#         if L == 0:
#             continue
#         gt_frames = gt_frames[:L]
#         pr_frames = pr_frames[:L]

#         # --- FID 特征 ---
#         real_feats = inception_extractor.get_features_from_images(gt_frames)
#         fake_feats = inception_extractor.get_features_from_images(pr_frames)
#         real_feats_all.append(real_feats)
#         fake_feats_all.append(fake_feats)

#         # --- LPIPS / Landmark / Expression ---
#         for g, f in zip(gt_frames, pr_frames):
#             lpips_metric.update(g, f)
#             landmark_metric.update(g, f)
#             expr_metric.update(g, f)

#     real_feats_all = np.concatenate(real_feats_all, axis=0)
#     fake_feats_all = np.concatenate(fake_feats_all, axis=0)

#     fid_value = compute_fid(real_feats_all, fake_feats_all)
#     lpips_value = lpips_metric.compute()
#     landmark_nme = landmark_metric.compute()
#     expr_consistency = expr_metric.compute()

#     metrics = {
#         "FID": fid_value,
#         "LPIPS": lpips_value,
#         "Landmark_NME": landmark_nme,
#         "Expression_Consistency": expr_consistency,
#         "Playback_FPS_avg": avg_playback_fps
#     }
#     return metrics


# # =========================
# # CLI
# # =========================

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gt_dir", type=str, required=True,
#                         help="GT（真实）视频所在目录，mp4")
#     parser.add_argument("--pred_dir", type=str, required=True,
#                         help="生成结果 mp4 目录")
#     parser.add_argument("--device", type=str, default="cuda",
#                         help='"cuda", "cuda:0" 或 "cpu"')
#     parser.add_argument("--out_json", type=str, default="metrics.json",
#                         help="输出指标 json 文件名")
#     args = parser.parse_args()

#     if args.device.startswith("cuda") and not torch.cuda.is_available():
#         print("[WARN] CUDA 不可用，自动切到 CPU")
#         args.device = "cpu"

#     t0 = time.time()
#     metrics = evaluate_all(args.gt_dir, args.pred_dir, device=args.device)
#     elapsed = time.time() - t0

#     print("\n====== Evaluation Results ======")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")
#     print(f"Total evaluation time: {elapsed:.2f} s")

#     with open(args.out_json, "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2, ensure_ascii=False)
#     print(f"\n[INFO] Metrics saved to {args.out_json}")


# if __name__ == "__main__":
#     main()


import os
import json
import argparse
import time

import numpy as np


def load_vertices_npy(path):
    """
    读取 .npy 顶点文件，返回 (T, N, 3) 数组：
    - T: 帧数
    - N: 顶点数
    - 3: (x,y,z)
    """
    arr = np.load(path)  # 可能是 (T, 3N) 或 (T, N, 3)

    if arr.ndim == 2:
        # (T, 3N) -> (T, N, 3)
        T, D = arr.shape
        assert D % 3 == 0, f"Illegal vert dim in {path}: {D}"
        N = D // 3
        arr = arr.reshape(T, N, 3)
    elif arr.ndim == 3:
        # 已经是 (T, N, 3)
        T, N, C = arr.shape
        assert C == 3, f"Last dim should be 3, got {C} in {path}"
    else:
        raise ValueError(f"Unsupported shape {arr.shape} in {path}")

    return arr  # (T, N, 3)


def compute_vertex_errors(gt_verts, pr_verts):
    """
    输入：
        gt_verts: (T, N, 3)
        pr_verts: (T, N, 3)
    输出：
        一个 dict，包含：
            - rmse_per_frame_mean
            - rmse_per_frame_std
            - rmse_normalized_mean
            - rmse_normalized_std
            - velocity_rmse_mean
            - acceleration_rmse_mean
    """

    # 对齐帧数和顶点数（取交集）
    T_gt, N_gt, _ = gt_verts.shape
    T_pr, N_pr, _ = pr_verts.shape

    T = min(T_gt, T_pr)
    N = min(N_gt, N_pr)

    gt = gt_verts[:T, :N, :]
    pr = pr_verts[:T, :N, :]

    # 1) 顶点 RMSE（每帧的 RMS，然后再对帧求平均 / 标准差）
    diff = gt - pr  # (T, N, 3)
    sq = np.square(diff)  # (T, N, 3)
    mse_per_frame = sq.mean(axis=(1, 2))  # (T,)
    rmse_per_frame = np.sqrt(mse_per_frame)  # (T,)

    rmse_per_frame_mean = float(rmse_per_frame.mean())
    rmse_per_frame_std = float(rmse_per_frame.std())

    # 2) 归一化 RMSE（除以第一帧 GT 包围盒对角线，类似 NME）
    gt_first = gt[0]  # (N,3)
    min_xyz = gt_first.min(axis=0)
    max_xyz = gt_first.max(axis=0)
    bbox_diag = np.linalg.norm(max_xyz - min_xyz) + 1e-8

    rmse_normalized = rmse_per_frame / bbox_diag
    rmse_norm_mean = float(rmse_normalized.mean())
    rmse_norm_std = float(rmse_normalized.std())

    # 3) 速度误差：v_t = x_{t+1} - x_t
    if T > 1:
        gt_v = gt[1:] - gt[:-1]   # (T-1,N,3)
        pr_v = pr[1:] - pr[:-1]
        v_diff = gt_v - pr_v
        v_rmse = np.sqrt(np.square(v_diff).mean(axis=(1, 2)))  # (T-1,)
        velocity_rmse_mean = float(v_rmse.mean())
    else:
        velocity_rmse_mean = None

    # 4) 加速度误差：a_t = v_{t+1} - v_t
    if T > 2:
        gt_v = gt[1:] - gt[:-1]  # (T-1,N,3)
        pr_v = pr[1:] - pr[:-1]
        gt_a = gt_v[1:] - gt_v[:-1]  # (T-2,N,3)
        pr_a = pr_v[1:] - pr_v[:-1]
        a_diff = gt_a - pr_a
        a_rmse = np.sqrt(np.square(a_diff).mean(axis=(1, 2)))
        acceleration_rmse_mean = float(a_rmse.mean())
    else:
        acceleration_rmse_mean = None

    metrics = {
        "Frames_gt": int(T_gt),
        "Frames_pred": int(T_pr),
        "Frames_used": int(T),
        "Num_vertices_gt": int(N_gt),
        "Num_vertices_pred": int(N_pr),
        "Num_vertices_used": int(N),
        "RMSE_per_frame_mean": rmse_per_frame_mean,
        "RMSE_per_frame_std": rmse_per_frame_std,
        "RMSE_normalized_mean": rmse_norm_mean,
        "RMSE_normalized_std": rmse_norm_std,
        "Velocity_RMSE_mean": velocity_rmse_mean,
        "Acceleration_RMSE_mean": acceleration_rmse_mean,
        "BBox_diag_first_frame": float(bbox_diag),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_npy",
        type=str,
        required=True,
        help="GT 顶点 .npy 路径，例如 data/vocaset/vertices_npy/FaceTalk_..._sentence01.npy"
    )
    parser.add_argument(
        "--pred_npy",
        type=str,
        required=True,
        help="预测结果 .npy 路径，例如 result/test_vocaset_....npy"
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="vert_metrics.json",
        help="输出指标 json 文件名"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gt_npy):
        raise FileNotFoundError(f"GT npy not found: {args.gt_npy}")
    if not os.path.isfile(args.pred_npy):
        raise FileNotFoundError(f"Pred npy not found: {args.pred_npy}")

    print("[INFO] GT   :", args.gt_npy)
    print("[INFO] Pred :", args.pred_npy)

    t0 = time.time()
    gt_verts = load_vertices_npy(args.gt_npy)
    pr_verts = load_vertices_npy(args.pred_npy)
    t1 = time.time()
    print(f"[INFO] Loaded npy in {t1 - t0:.3f} s")
    print(f"[INFO] GT shape   : {gt_verts.shape}  (T, N, 3)")
    print(f"[INFO] Pred shape : {pr_verts.shape}  (T, N, 3)")

    metrics = compute_vertex_errors(gt_verts, pr_verts)
    t2 = time.time()

    print("\n====== Vertex-space Metrics ======")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Total compute time: {t2 - t1:.3f} s")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Metrics saved to {args.out_json}")


if __name__ == "__main__":
    main()
