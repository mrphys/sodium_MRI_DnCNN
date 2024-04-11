Rapid 2D 23Na MRI of the calf using a denoising convolutional neural network
============================================================================

Rebecca R. Baker, Vivek Muthurangu, Marilena Rega, Stephen B. Walsh, Jennifer A. Steeden

Synopsis:
---------
A modified denoising convolutional neural network \[1\] is trained using 1H DICOM data from the fastMRI dataset \[2\] for application to 23Na MRI of the calf. Provided code includes model training and pretrained models as implemented for the paper. The ethics does not allow sharing of medical image data, thus 23Na data are not included. The fastMRI knee dataset required for training can be downloaded from https://fastmri.med.nyu.edu/.

Example
-------
![alt text](https://github.com/mrphys/sodium_MRI_DnCNN/blob/570a78e0fbfc7492dc2b7007ece50926604e2959/Pretrained_test_image_30NSA.png)

Installation and use
====================
For installation please:
1. Download github repository
2. From within the project folder, create Docker image and launch interactive docker container:
```
docker compose up --build -d
```
3. Dowload the fastMRI knee DICOM dataset from https://fastmri.med.nyu.edu/ and save the folder of DICOM files "knee_mri_clinical_seq_batch2" in the data folder
4. Test training by using the following command:
```
nohup docker compose exec tensorflow python train_network.py -m > training.log &
```
5. Shutdown docker container
```
docker compose down
```
0. Alternatively, can be used with VScode (.devcontainer folder) for development within the docker container

Note that only Linux is supported.

Trained models are saved in ./model/trained_models.

Pretrained models can be found in ./model/pretrained_models.

References
================
\[1\] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE Trans Image Process 2017;26:3142–55. https://doi.org/10.1109/TIP.2017.2662206

\[2\] Knoll F, Zbontar J, Sriram A, Muckley MJ, Bruno M, Defazio A, et al. fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning. Radiol Artif Intell 2020;2:e190007


