# This directory contains the training data for the [IS2021 ADReSSo Challenge](https://edin.ac/3p1cyaI).

The dataset contains enhanced, volume-normalised audio for the
diagnosis and MMSE score prediction tasks, and a table of MMSE scores
for model training (`adresso-train-mmse-scores.csv`). The abbreviation
`cn` denotes control patients, and `ad` denotes patients with a
(probable) Alzheimer's dementia diagnosis.

**Please note**: the original ADReSSo21 dataset can only be obtained by
applying on its hosting platform (https://talkbank.org/dementia/ADReSSo-2021/index.html).
This repository also provides locally augmented audio generated using
voice conversion; related code will be released later.

Below are the two directory layouts described together for clarity:

Official ADReSSo21 dataset layout

```text
ADReSSo21/diagnosis/
└── train/
    ├── audio/
    │   ├── ad/
    │   └── cn/
    └── segmentation/
        ├── ad/
        └── cn/
```

Locally augmented dataset (provided with this repository)

```text
ADReSSo21/diagnosis/
├── train/
│   ├── audio/
│   │   ├── ad/
│   │   └── cn/
└── train_aug/
    ├── audio/
    │   ├── ad/
    │   └── cn/
```

Also included are the utterance segmentation files (diarisation), in
CSV format. These files are for those who choose to do the segmented
prediction sub-task. The segmented prediction and speech-only
sub-tasks will be assessed separately.

Note: the locally augmented data included here contain audio files only (no transcription `text/` directories).
