# Towards a unified view of unsupervised non-local methods for image denoising: the NL-Ridge approach
Sébastien Herbreteau and Charles Kervrann

## Requirements

Here is the list of libraries you need to install to execute the code:
* Python 3.8
* Pytorch 1.10.0
* Numpy 1.21.2
* PIL 8.2.0

## Results

### Gray denoising
The average PSNR (dB) results of different methods on various datasets corrupted with Gaussian noise (sigma=15 and 25). Best performance among each category (unsupervised or supervised*) is in bold.

| Methods | Set12 | BSD68 | Urban100 |
|---------|:-------:|:--------:|:--------:|
| Noisy     |  24.61 / 20.17 |  24.61 / 20.17 |   24.61 / 20.17  |
| BM3D      | 32.37 / 29.97 | 31.07 / 28.57  | 32.35 / 29.70 |
| NL-Bayes  |   32.25 / 29.88  | 31.16 / **28.70** |  31.96 / 29.34 |
| **NL-Ridge**  |  **32.46** / 30.00  | **31.20** / 28.67  | **32.53** / **29.90** |
| DIP  |   30.12 / 27.54 | 28.83 / 26.59  |  - / - |
| Noise2Self  |   31.01 / 28.64 | 29.46 / 27.72 |  - / - |
| Self2Self  |   32.07 / **30.02** | 30.62 / 28.60 |  - / - |
|DnCNN*| **32.86** / **30.44** | **31.73** / **29.23** | 32.68 / 29.97|
|FFDnet*  |   32.75 / 30.43 | 31.63 / 29.19 | 32.43 / 29.92|
| LIDIA*  |  32.85 / 30.41 |  31.62 / 29.11 | **32.80** / **30.12** |



The average PSNR (dB) results of NL-Ridge on Set12 dataset corrupted with additive white Gaussian noise.

| $\sigma$ |  2 | 5 | 10 | 15 | 20 | 25 | 35 | 50 |
|---------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|  PSNR | 43.81 | 38.19 | 34.50 | 32.46 | 31.06 | 30.00 | 28.41 |  26.72 |

### Complexity
We want to emphasize that  NL-Ridge is relatively fast. We report here the execution times of different algorithms. It is
provided for information purposes only, as the implementation, the language used and the machine on which the code is run, highly influence the  results. The CPU used is a 2,3 GHz Intel Core i7 and the GPU is a GeForce RTX 2080 Ti. NL-Ridge has been entirely written in Python with Pytorch so it can run on GPU unlike its traditional counterparts. 


Running time (in seconds) of different methods on images of size 256x256. Run times are given on CPU and GPU if available.

| | BM3D | NL-Bayes | NL-Ridge | Self2Self | DnCNN | LIDIA |
|---------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|  CPU | 1.68 | 0.21 | 2.37 | n/a | 0.87 | 21.08|
|  GPU | n/a | n/a | 0.162 | 3877 | 0.007 | 1.184|
## Demo

To denoise an image with NL-Ridge (remove ``--add_noise`` if it is already noisy):
```
python ./demo.py --sigma 15 --add_noise --in ./datasets/Set12/09.png --out ./out/denoised.png
```

## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).

 

