# Time_Reversal_Fusion
<p align="center"> 
  <img width="1317" alt="TRF_teaser_figure" src="https://github.com/user-attachments/assets/6f793e43-afd8-4830-93f1-2c0171d8328a">
</p>

This is the official Pytorch implementation of Time Reversal Fusion (accepted at ECCV2024). 
We proposed a new sampling strategy called Time-Reversal Fusion (TRF) [8], which enables the image-to-video model to generate sequences toward a given end frame without any tuning or back-propagated optimization. We define this new task as "Bounded Generation" and it generalizes three scenarios in computer vision: 
  1) Generating subject motion with the two bound images capturing a moving subject. 
  2) Synthesizing camera motion using two images captured from different viewpoints of a static scene.
  3) Achieving video looping by using the same image for both bounds.

Please refer to the [arXiv paper](https://arxiv.org/abs/2403.14611) for more details.

  https://github.com/user-attachments/assets/b984c57c-a450-4071-996c-dc3df1445e79

## Todo
- [x] TRF code release
- [ ] TRF++ (Domain specific lora patch for downstream tasks) release

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/HavenFeng/Time_Reversal/
  cd Time_Reveral
  ```

### Requirements
* Python 3.10 (numpy, skimage, scipy, opencv)
* Diffusers
* PyTorch >= 1.8 (Diffusers compatible)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  If you encountered errors when installing Diffusers, please follow the [official installation guide](https://huggingface.co/docs/diffusers/en/installation) to re-install the library.

### Usage
1. **Run inference with samples in paper**  
    ```bash
    python svd_sequential_re.py 
    ```   
2. **TRF++ (add "lora" patches to enhance domain-specific task)**


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
More domain-specific lora patch models will be released soon

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/HavenFeng/TRUST/blob/main/LICENSE) file.
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/HavenFeng/TRUST/blob/main/LICENSE). 

## Acknowledgements
We would like to thank recent baseline works that allow us to easily perform quantitative and qualitative comparisons :)  
[FILM](https://github.com/soubhiksanyal/RingNet), 
[Wide-Baseline](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/renderer/rasterize_triangles.py), 
[Text2Cinemagraph](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/renderer/rasterize_triangles.py), 

This work was partly supported by the German Federal Ministry of Education and Research (BMBF): Tuebingen AI Center, FKZ: 01IS18039B
