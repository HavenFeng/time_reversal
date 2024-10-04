# Time_Reversal_Fusion
<p align="center"> 
  <img width="1317" alt="TRF_teaser_figure" src="https://github.com/user-attachments/assets/6f793e43-afd8-4830-93f1-2c0171d8328a">
</p>

This is the official Pytorch implementation of Time Reversal Fusion (accepted at ECCV2024). 
We proposed a new sampling strategy called Time-Reversal Fusion (TRF), which enables the image-to-video model to generate sequences toward a given end frame without any tuning or back-propagated optimization. We define this new task as "Bounded Generation" and it generalizes three scenarios in computer vision: 
  1) Generating subject motion with the two bound images capturing a moving subject. 
  2) Synthesizing camera motion using two images captured from different viewpoints of a static scene.
  3) Achieving video looping by using the same image for both bounds.

Please refer to the [arXiv paper](https://arxiv.org/abs/2403.14611) for more technical details and [Project Page](time-reversal.github.io) for more video results.

## Todo
- [x] TRF code release
- [ ] Bounded Generation Dataset release
- [ ] TRF++ (Domain specific lora patch for downstream tasks) release
- [ ] Gradio demo

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/HavenFeng/time_reversal/
  cd time_reveral
  ```

### Requirements
* Python 3.10 (numpy, skimage, scipy, opencv)
* Diffusers
* PyTorch >= 2.0.1 (Diffusers compatible)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  If you encountered errors when installing Diffusers, please follow the [official installation guide](https://huggingface.co/docs/diffusers/en/installation) to re-install the library.

### Usage
1. **Run inference with samples in paper**  
    ```bash
    python svd_sequential_re.py multiview
    ```
    Check different task results with "multiview", "video frames", "gym_motion" and "image2loop", the generated results can be found in the ./output folder.
2. **TRF++ (add LoRA "patches" to enhance domain-specific task)**  
    TRF was designed to probe SVD's bounded generation capabilities without fine-tuning, but we've observed SVD's biases in subject and camera motion, as well as sensitivity to conditioning factors like FPS and motion intensity. These required careful parameter tuning for different inputs. To improve generation quality and robustness for other downstream tasks, we fine-tuned LoRA "patch" on various domain-specific datasets, better supporting long-range linear motion and extreme 3D views generation.
   ```
    coming soon
   ```

## Evaluation
We evaluate our methods with the [Bounded Generation Dataset](https://time-reversal.github.io) compared to the domain-specific state-of-the-art methods.  
For more details of the evaluation, please check our [arXiv paper](https://arxiv.org/abs/2403.14611). 


## Citation
If you find our work useful to your research, please consider citing:
```
@inproceedings{Feng:TRF:ECCV2024,
  title = {Explorative In-betweening of Time and Space}, 
  author = {Feng, Haiwen and Ding, Zheng and Xia, Zhihao and Niklaus, Simon and Abrevaya, Victoria and Black, Michael J. and Zhang Xuaner}, 
  booktitle = {European Conference on Computer Vision}, 
  year = {2024}
}
```

## Notes
The video form of of our teaser image:

  https://github.com/user-attachments/assets/b984c57c-a450-4071-996c-dc3df1445e79

More domain-specific lora patch models will be released soon

## License
This code and model are available for non-commercial scientific research purposes.

## Acknowledgements
We would like to thank recent baseline works that allow us to easily perform quantitative and qualitative comparisons :)  
[FILM](https://github.com/google-research/frame-interpolation), 
[Wide-Baseline](https://github.com/yilundu/cross_attention_renderer), 
[Text2Cinemagraph](https://github.com/text2cinemagraph/text2cinemagraph/tree/master), 

This work was partly supported by the German Federal Ministry of Education and Research (BMBF): Tuebingen AI Center, FKZ: 01IS18039B
