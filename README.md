# SynVD-Net
The contribution of the proposed SynVD-Net is the proposed loss function deliberately derived for quality enhancement of synthesized view, especially considering its flickering distortion in synthesized video. The baseline could be the existing denoising CNN-based methods. UNet (adapted from CBDNet [1]), DnCNN [2], and RDN [3] are chosen as the baseline methods, and the associated SynVD-Nets are SynVD-UNet, SynVD-DnCNN, and SynVD-RDN. The objective and subjective results demonstrate that the proposed SynVD-Net is effective in quality enhancement of synthesized view. In addition, other conventional methods, i.e. BM3D [4], and VBM4D [5], and deep learning-based methods, SynIQE [6], and TSAN [7], are also adopted as comparison. (TSAN results are not published in our paper)

## Experimental results:
Objective comparison:
![image](https://user-images.githubusercontent.com/57431959/165043311-887da294-2d21-4b70-b6bd-d52e32877362.png)

Subjective comparison:

More results can be watched on website .

## Code usage:

### Requirements:
Pytorch 

### Train:
     #For training, the ground truth and distorted images training dataset should be arranged as
     /training-gt
     /training-input
     
     #Prepare training dataset and model training
     Python train -preprocess [True or False] -outf [model save path]

### Test
      #filepath of test images [ground truth, distorted images, and denoised images] 
    
      Examples:
      test_files_gt = 'lovebird1/gt/*.bmp'
      test_files = 'lovebird1/textureQP34_depthQP44/*.bmp'
      save_file = 'test/3/lovebird1'
      
      #testing
      Python test -logdir [pretained model path] 

### Other issues:
  For SynIQE method, neighboring view images are taken as input. For more details, please refer to the paper [6].


## References:
* [1] CBDNet: S. Guo, Z. Yan, K. Zhang, W. Zuo, and L. Zhang, “Toward convolutional blind denoising of real photographs,” in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, Jun. 2019, pp. 1712–1722.
* [2] DnCNN: K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, Jul. 2017. 
* [3] RDN: Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu, “Residual dense network for image restoration,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 7, pp. 2480–2495, Jul. 2021.
* [4] BM3D: K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image denoising by sparse 3-D transformdomain collaborative filtering,” IEEE Trans. Image Process., vol. 16, no. 6, pp. 2080–2095, Sept. 2007.
* [5] VBM4D: M. Maggioni, G. Boracchi, A. Foi, and K. Egiazarian, “Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms,” IEEE Trans. Image Process., vol. 21, no. 9, pp. 3952–3966, Sept. 2012.
* [6] SynIQE: L. Zhu, Y. Zhang, S. Wang, H. Yuan, S. Kwong, and H. H. -S. Ip, “Convolutional neural network-based synthesized view quality enhancement for 3D video coding,” IEEE Trans. Image Process., vol. 27, no. 11, pp. 5365–5377, Nov. 2018.
* [7] TSAN: Z. Pan, W. Yu, J. Lei, N. Ling, and S. Kwong, "TSAN: synthesized view quality enhancement via two-stream attention network for 3D-HEVC," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 1, pp. 345-358, Jan. 2022.

If you find this work is useful, you can cite 'H. Zhang, Y. Zhang, L. Zhu and W. Lin, "Deep Learning-based Perceptual Video Quality Enhancement for 3D Synthesized View," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2022.3147788.' in your work.

