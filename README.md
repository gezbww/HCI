##
## Environment

- Linux and Windows (tested on Windows 10 and 11)
- Python 3.9+
- PyTorch 1.10.1+cu111

## Dependencies

- [ffmpeg](https://www.ffmpeg.org/download.html)
- Check the required python packages and libraries in `requirements.txt`.
- Install them by running the command: `pip install -r requirements.txt`

## Data
### VOCASET

Download the training data from: https://voca.is.tue.mpg.de/download.php.

Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `data/vocaset/`.
Read the downloaded data and convert it to .npy and .wav format accepted by the model. Run the following instructions for this:

```commandline
cd data/vocaset
python process_voca_data.py
```
## Model Training 
### Training and Testing

| Arguments     | BIWI  | VOCASET | Multiface | UUDaMM | BEAT |
|---------------|-------|---------|-----------|--------|------|
| --dataset     |  BIWI | vocaset | multiface |  damm  | beat |
| --vertice_dim | 70110 |  15069  |   18516   |   192  |  51  |
| --output_fps  |   25  |    30   |     30    |   30   |  30  |

- Train the model by running the following command:
	```
	python main.py
	```
	The test split predicted results will be saved in the `result/`. The trained models (saves the model in 25 epoch interval) will be saved in the `save/` folder.


### Predictions

- Download the trained weights from [here](https://mega.nz/folder/jlBF0Dpa#U3G1lJCZ4dijMoSc9gmqSg) and add them to the folder `pretrained_models`.
- To generate predictions use the commands:
Vocaset
```commandline
python predict.py --dataset vocaset --vertice_dim 15069 --feature_dim 256 --output_fps 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" --model_name "pretrained_vocaset" --fps 30 --condition "FaceTalk_170728_03272_TA" --subject "FaceTalk_170731_00024_TA" --diff_steps 1000 --gru_dim 256 --wav_path "test.wav"
```
### Acknowledgements

We borrow and adapt the code from([https://github.com/galib360/FaceXHuBERT](https://github.com/uuembodiedsocialai/FaceDiffuser)), 

Thanks for making their code available and facilitating future research.
Additional thanks to [huggingface-transformers](https://huggingface.co/) for the implementation of HuBERT.

We are also grateful for the publicly available datasets used during this project:
- MPI-IS for releasing the VOCASET dataset. 


Any third-party packages are owned by their respective authors and must be used under their respective licenses.
