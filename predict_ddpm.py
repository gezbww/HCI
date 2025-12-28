
import librosa
import argparse
import torch
import trimesh
import numpy as np
import pandas as pd
import cv2
import os
import ffmpeg
import gc
import pyrender
import pickle
import time

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from transformers import Wav2Vec2Processor
from utils import *


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

 
    if args.dataset == 'beat':
        model = FaceDiffBeat(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers
        )
    elif args.dataset == 'damm_rig_equal':
        model = FaceDiffDamm(args)
    else:
        model = FaceDiff(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers
        )


    ckpt_path = os.path.join('pretrained_models', '{}.pth'.format(args.model_name))
    state_dict = torch.load(ckpt_path, map_location='cpu')  
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()


    if args.low_mem and args.dataset.lower() == 'vocaset':
        model_device = torch.device('cpu')
        diff_device = torch.device('cpu')
        use_half = False          # CPU 不用 half
        print("[INFO] vocaset + low_mem: 全流程在 CPU 上采样，几乎不占 GPU 显存。")
    else:
        model_device = torch.device(args.device)
        diff_device = model_device
        use_half = (model_device.type == 'cuda' and torch.cuda.is_available())
        if use_half:
            print("[INFO] 使用 GPU + FP16 进行推理。")

    model = model.to(model_device)
    if use_half:
        model = model.half()
    model.eval()


    template_file = os.path.join(args.data_path, args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    one_hot_labels = np.eye(len(train_subjects_list))

    idx = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[idx]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.tensor(one_hot, dtype=torch.float32)

    if use_half:
        one_hot = one_hot.half()
    one_hot = one_hot.to(diff_device)

    if args.dataset in ["BIWI", "multiface", "vocaset"]:
        temp = templates[args.subject]
    else:
        temp = np.zeros((args.vertice_dim // 3, 3))

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.tensor(template, dtype=torch.float32)
    if use_half:
        template = template.half()
    template = template.to(diff_device)

 
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    start_time = time.time()

    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)


    processor = Wav2Vec2Processor.from_pretrained("pretrained_models/hubert-xlarge-ls960-ft")
    audio_feature = processor(
        speech_array,
        return_tensors="pt",
        padding="longest",
        sampling_rate=sampling_rate
    ).input_values  

    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
    if use_half:
        audio_feature = audio_feature.half()
    audio_feature = audio_feature.to(diff_device)

  
    diffusion = create_gaussian_diffusion(args)

  
    num_frames = int(audio_feature.shape[0] / sampling_rate * args.output_fps)
    num_frames -= 1


    with torch.no_grad():
        if use_half and diff_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = diffusion.plms_sample_loop(
                    model.to(diff_device),                     #
                    (1, num_frames, args.vertice_dim),
                    num_steps=args.num_steps,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio_feature,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=args.skip_steps,
                    init_image=None,
                    progress=True,
                    noise=None,
                    device=str(diff_device)                    # 
                )
        else:

            model = model.to(diff_device)
            prediction = diffusion.plms_sample_loop(
                model,
                (1, num_frames, args.vertice_dim),
                num_steps=args.num_steps,
                clip_denoised=False,
                model_kwargs={
                    "cond_embed": audio_feature,
                    "one_hot": one_hot,
                    "template": template,
                },
                skip_timesteps=args.skip_steps,
                init_image=None,
                progress=True,
                noise=None,
                device=str(diff_device)
            )

    prediction = prediction.squeeze()


    del model, diffusion, audio_feature, one_hot, template
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    prediction = prediction.detach().float().cpu().numpy()

    if args.dataset == 'damm_rig_equal':
        with open('data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
            RIG_SCALER = pickle.load(f)
        prediction = RIG_SCALER.inverse_transform(prediction)

    elapsed = time.time() - start_time
    print("Inference time for ", prediction.shape[0], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / prediction.shape[0], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * args.fps) / prediction.shape[0]), " seconds.")

    out_file_name = test_name + "_" + args.dataset + "_" + args.subject + "_condition_" + args.condition
    np.save(os.path.join(args.result_path, out_file_name), prediction)

   
    if args.dataset == 'damm_rig_equal':
        df = pd.DataFrame(prediction)
        df.to_csv(os.path.join(args.result_path, f"{out_file_name}_Damm.csv"), index=None, header=None)


def render(args):
    fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    render_path = "demo/renders/"
    frames_folder = render_path + "tmp/"
    video_woA_folder = frames_folder
    video_wA_folder = render_path + "video_with_audio/"

    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    if not os.path.exists(video_wA_folder):
        os.makedirs(video_wA_folder)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    out_file_name = test_name + "_" + args.dataset + "_" + args.subject + "_condition_" + args.condition
    predicted_vertices_path = os.path.join(args.result_path, out_file_name + ".npy")
    template_file = f"data/{args.dataset}/templates/face_template.obj"
    camera_dist = 2.0 if args.dataset == "multiface" else 1.0

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, camera_dist],
                            [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(640, 480)

    print("rendering the predicted sequence: ", test_name)
    video_woA_path = video_woA_folder + out_file_name + '.mp4'
    video_wA_path = video_wA_folder + out_file_name + '.mp4'
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    ref_mesh = trimesh.load_mesh(template_file, process=False)
    seq = np.load(predicted_vertices_path)
    seq = np.reshape(seq, (-1, args.vertice_dim // 3, 3))
    ref_mesh.vertices = seq[0, :, :]

    for f in range(seq.shape[0]):
        ref_mesh.vertices = seq[f, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
        scene = pyrender.Scene()
        scene.add(py_mesh)
        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)

        output_frame = os.path.join(frames_folder, f"{f:04d}.png")
        cv2.imwrite(output_frame, color)
        frame = cv2.imread(output_frame)
        video.write(frame)

    video.release()

    input_video = ffmpeg.input(video_woA_path)
    input_audio = ffmpeg.input(wav_path)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()

    del video, seq, ref_mesh, r
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pretrained_BIWI")
    parser.add_argument("--data_path", type=str, default="data", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--dataset", type=str, default="BIWI", help='eg: BIWI / vocaset / multiface')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
    parser.add_argument("--vertice_dim", type=int, default=70110, help='BIWI: 23370*3')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help='e.g. "cuda:0", "cuda:1" 或 "cpu"（非 vocaset+low_mem 的默认设备）')
    parser.add_argument("--train_subjects", type=str,
                        default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--test_subjects", type=str,
                        default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav",
                        help='path of the input audio signal in .wav format')
    parser.add_argument("--result_path", type=str, default="demo/result",
                        help='path of the predictions in .npy format')
    parser.add_argument("--condition", type=str, default="M3",
                        help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1",
                        help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--template_path", type=str, default="templates.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates",
                        help='path of the mesh in BIWI topology')
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--emotion", type=int, default=1,
                        help='style control for emotion, 1 for expressive animation, 0 for neutral animation')
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--gru_dim", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--skip_steps", type=int, default=0)
    parser.add_argument(
    "--no_render",
    action="store_true",
    help="只做预测"
)

    parser.add_argument("--low_mem", action="store_true",
                        help="避免 GPU OOM。")

    args = parser.parse_args()

    test_model(args)

    if args.dataset in ["BIWI", "multiface", "vocaset"] and (not args.no_render):
        render(args)


if __name__ == "__main__":
    main()
