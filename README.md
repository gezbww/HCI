##
##Demo Results
###Here are visual comparisons of the four diffusion methods implemented in this project:
## 🎥 效果演示

这里是四个不同模型的生成效果展示：

<div align="center">
  <table>
    <tr>
      <td align="center">
        <video width="280" height="280" controls>
          <source src="PLMS.mp4" type="video/mp4">
          您的浏览器不支持 video 标签。
        </video>
        <br>
        <strong>PLMS</strong>
      </td>
      <td align="center">
        <video width="280" height="280" controls>
          <source src="DDPM.mp4" type="video/mp4">
          您的浏览器不支持 video 标签。
        </video>
        <br>
        <strong>DDPM</strong>
      </td>
    </tr>
    <tr>
      <td align="center">
        <video width="280" height="280" controls>
          <source src="DDIM.mp4" type="video/mp4">
          您的浏览器不支持 video 标签。
        </video>
        <br>
        <strong>DDIM</strong>
      </td>
      <td align="center">
        <video width="280" height="280" controls>
          <source src="dpm_solver.mp4" type="video/mp4">
          您的浏览器不支持 video 标签。
        </video>
        <br>
        <strong>dpm_solver</strong>
      </td>
    </tr>
  </table>
</div>

> 📌 提示：点击视频右下角的全屏按钮可获得更好的观看体验。

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
