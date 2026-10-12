## 데이터 폴더 구조

`python -m preprocess.mpii`를 실행하면 `data/` 아래에 다음이 생긴다. `data/`는 `.gitignore`에 들어 있어 저장소에 올라가지 않는다. 변환 과정은 [preprocess/README.md](preprocess/README.md)에 정리되어 있다.

```
data/
├── raw/
│   └── MPIIGaze.tar.gz                  # 원본 아카이브 (2.0G). 있으면 다시 받지 않는다
├── MPIIGaze/                            # 압축 해제한 원본 (5.2G)
│   ├── ReadMe.txt
│   ├── 6 points-based face model.mat
│   ├── Annotation Subset/p00.txt … p14.txt
│   ├── Data/
│   │   ├── Original/p00 … p14/          # 전처리 입력 (4.4G)
│   │   │   ├── Calibration/             # Camera.mat, monitorPose.mat, screenSize.mat
│   │   │   └── day01 … dayNN/           # 0001.jpg … (웹캠 프레임) + annotation.txt
│   │   └── Normalized/p00 … p14/        # day01.mat … (데이터셋 제공 정규화본, 쓰지 않음)
│   └── Evaluation Subset/
│       ├── ReadMe.txt
│       ├── annotation for face image/p00.txt … p14.txt
│       └── sample list for eye image/p00.txt … p14.txt   # 변환할 샘플 목록
└── EyeBased/
    └── MPIIGaze/                        # 전처리 결과 (190M), 학습에 쓰는 데이터
        ├── Image/p00 … p14/1.jpg … 3000.jpg   # 36×60 grayscale 눈 이미지
        └── Label/p00.label … p14.label        # 샘플당 한 줄, 첫 줄은 헤더
```

- 전처리가 읽는 원본은 `Data/Original`의 `Calibration/Camera.mat`(카메라 내부 행렬), 각 day의 `annotation.txt`와 프레임, 그리고 `Evaluation Subset/sample list for eye image`의 목록뿐이다. 나머지는 데이터셋에 함께 들어 있는 파일이다.
- day 폴더 수는 사람마다 다르다 (7~69개).
- 결과는 사람당 3,000개(왼눈 1,500 + 오른눈 1,500), 15명 합계 45,000개다. 오른눈 이미지는 좌우 반전되어 왼눈처럼 저장된다.
- 이미지 파일 이름은 목록 순서대로 붙인 일련번호이며, 원본 프레임 위치는 라벨의 `Origin` 칸에 남는다.

라벨 예 (`Label/p00.label`, 소수는 줄임):

```
Image Origin WhichEye 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin
p00/1.jpg day08/0069.jpg left -0.507,-0.112,-0.029 0.247,-0.052 -0.115,0.508 -0.023,-0.171,-0.037 1.0,1.0,1.205 0.0,0.0,600.0
```

`reader.py`는 `loader("data/EyeBased/MPIIGaze/Label/p00.label", "data/EyeBased/MPIIGaze/Image")`처럼 라벨 파일과 `Image/` 폴더를 받는다. 각 칸의 의미는 [preprocess/README.md의 5. 출력](preprocess/README.md#5-출력)을 본다.


## 과제




## Installation (conda)

```
conda create -n gaze python=3.12 -y
conda activate gaze

conda install -c conda-forge jupyter matplotlib # for visualization and Eda
```
