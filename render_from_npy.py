import os
import gc
import argparse

import numpy as np
import cv2
import ffmpeg


def render_pointcloud_from_npy(args):
    # 当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    npy_path = os.path.join(base_dir, args.npy_name)
    wav_path = os.path.join(base_dir, args.wav_name) if args.wav_name is not None else None

    render_root = os.path.join(base_dir, "renders_pointcloud")
    os.makedirs(render_root, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    video_path = os.path.join(render_root, base_name + "_pc_noaudio.mp4")
    video_with_audio_path = os.path.join(render_root, base_name + "_pc_withaudio.mp4")

    print("========== 路径检查 ==========")
    print("[INFO] base_dir      :", base_dir)
    print("[INFO] npy_path      :", npy_path)
    print("[INFO] wav_path      :", wav_path)
    print("[INFO] video(no)     :", video_path)
    print("[INFO] video(with)   :", video_with_audio_path)
    print("================================")

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"找不到 npy 文件: {npy_path}")
    if wav_path is not None and not os.path.isfile(wav_path):
        print(f"[WARN] 找不到 wav 文件: {wav_path}，将只导出无音频视频。")
        wav_path = None

    # ====== 读取顶点序列 ======
    data = np.load(npy_path)  # shape: (T, vertice_dim)
    T, vertice_dim = data.shape
    num_verts = vertice_dim // 3
    print(f"[INFO] data shape: {data.shape}, num_verts = {num_verts}")

    # reshape 为 (T, N, 3)
    data = data.reshape(T, num_verts, 3)

    # 用 x, y 分量做 2D 投影
    xs = data[:, :, 0]
    ys = data[:, :, 1]

    # 全局归一化范围
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    print(f"[INFO] x range: [{min_x:.4f}, {max_x:.4f}]")
    print(f"[INFO] y range: [{min_y:.4f}, {max_y:.4f}]")

    # 图像大小
    W, H = args.width, args.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, args.fps, (W, H))

    # 避免除零
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    print("[INFO] 开始渲染散点云...")

    for t in range(T):
        img = np.zeros((H, W, 3), dtype=np.uint8)  # 黑底

        x = xs[t]
        y = ys[t]

        # 归一化到 [0,1]
        nx = (x - min_x) / range_x
        ny = (y - min_y) / range_y

        # 留一点边界：0.1 ~ 0.9
        nx = 0.1 + 0.8 * nx
        ny = 0.1 + 0.8 * ny

        # 转成像素坐标（注意 y 轴翻转）
        px = (nx * (W - 1)).astype(np.int32)
        py = ((1.0 - ny) * (H - 1)).astype(np.int32)

        # 画点：白色小圆点
        for i in range(num_verts):
            cv2.circle(img, (px[i], py[i]), args.radius, (255, 255, 255), -1)

        # 写这一帧到视频
        video.write(img)

        if (t + 1) % 50 == 0 or t == T - 1:
            print(f"[INFO] rendered frame {t + 1}/{T}")

    video.release()
    print("[INFO] 无音频视频已保存:", video_path)

    # ====== 合成音频（如果有 wav） ======
    if wav_path is not None:
        print("[INFO] 使用 ffmpeg 合成音频...")
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(wav_path)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_with_audio_path).run()
        print("[INFO] 带音频视频已保存:", video_with_audio_path)
    else:
        print("[INFO] 未提供 wav，跳过音频合成。")

    gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npy_name",
        type=str,
        required=True,
        help="当前目录下的 npy 文件名，例如 xxx_vocaset_....npy"
    )
    parser.add_argument(
        "--wav_name",
        type=str,
        default=None,
        help="当前目录下的 wav 文件名（可选，不填则不加音频）"
    )
    parser.add_argument(
        "--vertice_dim",
        type=int,
        default=15069,   # vocaset: 5023*3 = 15069
        help="顶点维度（vocaset = 5023*3 = 15069; BIWI=23370*3=70110）"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="视频帧率"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="视频宽度"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="视频高度"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="散点的半径（像素）"
    )

    args = parser.parse_args()
    render_pointcloud_from_npy(args)


if __name__ == "__main__":
    main()
