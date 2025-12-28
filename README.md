##
##Demo Results
###Here are visual comparisons of the four diffusion methods implemented in this project:
## 🎥 效果演示

这里是四个不同模型的生成效果展示：

<h2 align="center">🎥 Demo Results</h2>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>PLMS</b><br>
        <video src="PLMS.mp4" width="100%" controls preload></video>
      </td>
      <td align="center">
        <b>DDPM</b><br>
        <video src="DDPM.mp4" width="100%" controls preload></video>
      </td>
    </tr>
    <tr>
      <td align="center">
        <b>DDIM</b><br>
        <video src="DDIM.mp4" width="100%" controls preload></video>
      </td>
      <td align="center">
        <b>dpm_solver</b><br>
        <video src="DPM_solver.mp4" width="100%" controls preload></video>
      </td>
    </tr>
  </table>
</div>

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
## different methods 
PLMS DDPM DDIM dpm_solver

### How to run these codes

- Download the trained weights from [here](https://mega.nz/folder/jlBF0Dpa#U3G1lJCZ4dijMoSc9gmqSg) and add them to the folder `pretrained_models`.
- To generate predictions use the commands:
Vocaset
```commandline
bash predict_ddpm.sh
```
### Acknowledgements

We borrow and adapt the code from([https://github.com/galib360/FaceXHuBERT](https://github.com/uuembodiedsocialai/FaceDiffuser)), 

Thanks for making their code available and facilitating future research.
Additional thanks to [huggingface-transformers](https://huggingface.co/) for the implementation of HuBERT.

We are also grateful for the publicly available datasets used during this project:
- MPI-IS for releasing the VOCASET dataset. 


Any third-party packages are owned by their respective authors and must be used under their respective licenses.
