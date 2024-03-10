# Towards a unified view of unsupervised non-local methods for image denoising: the NL-Ridge approach
Sébastien Herbreteau and Charles Kervrann

## Requirements

Here is the list of libraries you need to install to execute the code:
* Python 3.11
* Pytorch 2.2
* Torchvision 0.17
* Einops 0.7.0

## Install

To install in an environment using pip:

```
python -m venv .nlridge_env
source .nlridge_env/bin/activate
pip install /path/to/NL-Ridge
```

## Demo

To denoise an image with NL-Ridge (remove ``--add_noise`` if it is already noisy):
```
python ./demo.py --sigma 15 --add_noise --in ./test_images/barbara.png --out ./denoised.png
```

Or use directly the Pytorch class NL-Ridge within your code:
```
m_nlridge = NLRidge() # instantiate the NL-Ridge class
y = 5 * torch.randn(1, 1, 100, 100) # image of pure Gaussian noise with variance 5^2
x_hat = m_nlridge(y, sigma=5, noise_type='gaussian-homoscedastic', constraints='affine', p1=7, p2=7, k1=18, k2=55, w=37, s=4) 
```
(see the meaning of the parameters in file nlridge.py, method set_parameters)


## Results

### Gray denoising
The average PSNR (dB) results of different methods on various datasets corrupted with Gaussian noise (sigma=15, 25 and 50). Best performance among each category is in bold. Second best is underlined.


<img src="https://user-images.githubusercontent.com/88136310/205092725-c1e93e06-8879-4ede-aa8d-a2bba311bdd9.jpeg" width="750">

The average PSNR (dB) results of NL-Ridge on Set12 dataset corrupted with additive white Gaussian noise.

| sigma |  2 | 5 | 10 | 15 | 20 | 25 | 35 | 50 |
|---------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|  PSNR | 43.81 | 38.19 | 34.50 | 32.46 | 31.06 | 30.00 | 28.41 |  26.73 |

### Complexity
We want to emphasize that  NL-Ridge is relatively fast. We report here the execution times of different algorithms. It is
provided for information purposes only, as the implementation, the language used and the machine on which the code is run, highly influence the  results. The CPU used is a 2,3 GHz Intel Core i7 and the GPU is a GeForce RTX 2080 Ti. NL-Ridge has been entirely written in Python with Pytorch so it can run on GPU unlike its traditional counterparts. 


Running time (in seconds) of different methods on images of size 256x256. Run times are given on CPU and GPU if available.

| | BM3D | NL-Bayes | NL-Ridge | Self2Self | DnCNN | LIDIA |
|---------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|  CPU | 1.68 | 0.21 | 0.66 | n/a | 0.35 | 21.08|
|  GPU | n/a | n/a | 0.162 | 3877 | 0.007 | 1.184|


## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).
