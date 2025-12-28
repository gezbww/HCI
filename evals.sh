# python evals.py \
#   --gt_dir /home/user/gzb/HCI/FaceDiffuser-main/renders_pointcloud/gt_videos \
#   --pred_dir /home/user/gzb/HCI/FaceDiffuser-main/renders_pointcloud/pred_videos \
#   --device cuda \
#   --out_json ddpm_plms_metrics.json

python evals.py \
  --gt_npy /home/user/gzb/HCI/FaceDiffuser-main/data/vocaset/vertices_npy/FaceTalk_170731_00024_TA_sentence01.npy \
  --pred_npy /home/user/gzb/HCI/FaceDiffuser-main/result/test_vocaset_FaceTalk_170731_00024_TA_condition_FaceTalk_170728_03272_TA.npy \
  --out_json vocaset_vert_metrics_sentence01.json