# python predict_plms.py \
#     --dataset vocaset \
#     --num_steps 10 \
#     --vertice_dim 15069 \
#     --feature_dim 256 \
#     --output_fps 30 \
#     --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
#     --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
#     --model_name "pretrained_vocaset" \
#     --fps 30 \
#     --condition "FaceTalk_170728_03272_TA"\
#     --subject "FaceTalk_170731_00024_TA"\
#     --diff_steps 1000 \
#     --gru_dim 256 \
#     --wav_path "test.wav" \

# python predict_plms.py \
#   --dataset vocaset \
#   --data_path data \
#   --vertice_dim 15069 \
#   --feature_dim 256 \
#   --output_fps 30 \
#   --fps 30 \
#   --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
#   --test_subjects  "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
#   --model_name "pretrained_vocaset" \
#   --condition "FaceTalk_170728_03272_TA" \
#   --subject   "FaceTalk_170731_00024_TA" \
#   --wav_path test.wav \
#   --result_path ./result \
#   --template_path templates.pkl \
#   --render_template_path templates \
#   --diff_steps 1000 \
#   --num_steps 50 \
#   --gru_dim 256 \
#   --gru_layers 2 \
#   --device cuda \
#   --device_idx 0 \
#   --input_fps 50 \
#   --emotion 1 \
#   --skip_steps 0

python predict_plms.py \
  --dataset vocaset \
  --data_path data \
  --vertice_dim 15069 \
  --feature_dim 256 \
  --output_fps 30 \
  --fps 30 \
  --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
  --test_subjects  "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
  --model_name "pretrained_vocaset" \
  --condition "FaceTalk_170728_03272_TA" \
  --subject   "FaceTalk_170731_00024_TA" \
  --wav_path test.wav \
  --result_path ./result \
  --template_path templates.pkl \
  --render_template_path templates \
  --diff_steps 1000 \
  --num_steps 50 \
  --gru_dim 256 \
  --gru_layers 2 \
  --device cuda:0 \
  --low_mem \
  --no_render




#python predict.py --dataset vocaset --vertice_dim 15069 --feature_dim 256 --output_fps 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" --model_name "pretrained_vocaset" --fps 30 --condition "FaceTalk_170728_03272_TA" --subject "FaceTalk_170731_00024_TA" --diff_steps 1000 --gru_dim 256 --wav_path "test.wav"