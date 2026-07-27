# I $`^3`$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation


## Overview

Multimodal recommender systems (MRS) improve recommendation performance by integrating diverse semantic information from multiple modalities. However, existing MRS methods often assume the availability of all modalities, which is frequently violated in real-world scenarios. Factors such as inaccessible product images, missing descriptions, and inconsistent user-generated content contribute to the widespread occurrence of missing modalities. These challenges significantly degrade the robustness and generalization capabilities of current models. To address these challenges, we introduce a novel method called I $^3$-MRec, which uses **I**nvariant learning with **I**nformation bottleneck principle for **I**ncomplete **M**odality **Rec**ommendation. In order to achieve robust performance in missing modality scenarios, I$^3$-MRec enforces two pivotal properties: (i) cross-modal preference invariance, which ensures consistent user preference modeling across varying modality environments, and (ii) compact yet effective modality representation, which filters out task-irrelevant modality information while maximally preserving essential features relevant to recommendation. By treating each modality as a distinct semantic environment, I $^3$-MRec employs invariant risk minimization (IRM) to learn robust, modality-specific user and item representations. Furthermore, a missing-aware fusion module grounded in the Information Bottleneck (IB) principle extracts compact and effective item embeddings by suppressing modality noise and preserving core user preference signals. Extensive experiments conducted on three real-world datasets demonstrate that I $^3$-MRec consistently outperforms existing state-of-the-art MRS methods across various modality-missing scenarios, highlighting its effectiveness and robustness in practical applications.
![architecture](./img/framework.png)

## Updates
- Update (November 24, 2025): Fixed the issue caused by the missing training data file, and added data preprocessing and generation functions based on intermediate files.
- Update (November 24, 2025): The [TikTok](https://drive.google.com/drive/folders/11wEn5k1Kzusj1GkdAlCcfS3GbBWGzFpX?usp=drive_link) dataset has been released. The dataset is sourced from and made publicly available by the [MILK](https://github.com/HaoyueBai98/MILK) work.


## Environment

- python==3.9
- pytorch==1.13.0
- numpy== 1.26.4
- numba==0.60.0


## Dataset

Download from Google Drive: [Baby/Clothing](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) from [MMRec](https://github.com/enoche/MMRec).
The data already contains text and image features extracted from Sentence-Transformers and CNN, which is provided by [MMRec](https://github.com/enoche/MMRec).
Please move your downloaded data into the 'Data' folder for model training.



## Training / Test for Missing Modality Setting on Baby
Note that: `--missing_rate` is defined as missing modality rate, modifying it to change the missing rate of modaliy feature for training data.

### Full Training Full Test:
```
python main.py --dataset baby --max_info_coeff 1e-3 --min_info_coeff 1e-5 --reg_coeff 1e-3 --penalty_coeff 300  --lr 1e-3  --exp_mode ff
```

### Full Training Missing Test:
```
python main.py --dataset baby --max_info_coeff 1e-3 --min_info_coeff 1e-5 --reg_coeff 1e-3 --penalty_coeff 300  --lr 1e-3 --missing_rate 0.3  --exp_mode fm
```
### Missing Training Missing Test (MTMT): 
```
python main.py --dataset baby --max_info_coeff 1e-3 --min_info_coeff 1e-5 --reg_coeff 1e-3 --penalty_coeff 300  --lr 1e-3 --missing_rate 0.3 --exp_mode mm
```

## Training / Test for Missing Modality Setting on Clothing

### Full Training Full Test:
```
python main.py --dataset clothing --max_info_coeff 1e-2 --min_info_coeff 1e-5 --reg_coeff 1e-2 --penalty_coeff 1  --lr 1e-2  --exp_mode ff
```

### Full Training Missing Test:
```
python main.py --dataset clothing --max_info_coeff 1e-2 --min_info_coeff 1e-6 --reg_coeff 1e-2 --penalty_coeff 1 --missing_rate 0.3 --lr 1e-2  --exp_mode fm
```
### Missing Training Missing Test (MTMT): 
```
python main.py --dataset clothing --max_info_coeff 1e-2 --min_info_coeff 1e-6 --reg_coeff 1e-2 --penalty_coeff 1 --missing_rate 0.3 --lr 1e-2  --exp_mode mm
```
