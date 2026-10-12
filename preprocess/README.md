# preprocess

[MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) 원본을 [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/) **eye-based** 형식(36×60 grayscale 눈 이미지 + `.label`)으로 바꾸는 코드다. 
정규화 방식은 [Sugano et al. 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Sugano_Learning-by-Synthesis_for_Appearance-based_2014_CVPR_paper.html), [Zhang et al. 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.html)의 perspective normalization이다.

| 파일 | 역할 |
|------|------|
| `mpii.py` | 다운로드, 압축 해제, person별 변환 루프, `.label` 쓰기 (CLI) |
| `mpii_norm.py` | 정규화 코어 `Norm`, 3D→2D 변환, 오른눈 flip 함수 |


```bash
# 저장소 루트에서
pip install numpy opencv-python scipy
python -m preprocess.mpii            # download + convert
```

---

## 1. 전체 흐름

```
download ─► data/raw/MPIIGaze.tar.gz ─► data/MPIIGaze/ (압축 해제)
                                              │
convert ◄─────────────────────────────────────┘
  for person in p00..p14:
    Camera.mat 읽기 (카메라 내부 행렬 K)
    day별 annotation.txt 전부 읽기
    for (day/img, eye) in "Evaluation Subset/sample list for eye image/pXX.txt":
      1. 원본 프레임을 grayscale로 읽기
      2. annotation 한 줄 해석
      3. 눈 중심 기준 시점 정규화 (Norm) → 60×36 warp
      4. histogram equalization
      5. 정규화 좌표계로 gaze / head / origin 변환
      6. 오른눈이면 좌우 반전
      7. 3D → 2D (yaw, pitch)
      8. Image/pXX/{n}.jpg, Label/pXX.label 한 줄 쓰기
  출력 검증 (첫 이미지가 36×60인지)
```

- `download`는 원본이 이미 풀려 있으면 건너뛰고, tar.gz가 있으면 다시 받지 않는다. 받는 중에는 `.part`로 저장했다가 끝나면 이름을 바꾸므로 중간에 끊겨도 깨진 아카이브가 남지 않는다.
- 변환 대상은 `Data/Original`의 프레임 중 **Evaluation Subset 목록에 있는 것만**이다. `Data/Normalized`는 쓰지 않는다.
  - 목록의 한 행이 샘플 하나(프레임 하나에서 잘라낸 눈 하나)다. 사람당 3,000행(`left` 1,500 + `right` 1,500)이고, 15명 합계 45,000 샘플이다.
  - p13 목록에는 같은 (프레임, 눈) 행이 중복된 경우가 4개 있다. 코드는 이를 걸러내지 않고 별도 이미지로 저장하므로, p13의 고유 샘플은 2,996개다.

---

## 2. annotation 해석 (`decode_annotation`)

목록의 한 줄은 `day03/0123.jpg left` 같은 형태다. 파일 이름의 번호(`0123` → 123)로 `day03/annotation.txt`의 **123번째 줄**(0-based 인덱스 122)을 꺼낸다.

한 줄은 공백으로 구분된 41개 숫자이며, 그중 다음만 쓴다 (0-based 슬라이스, 단위 mm, 카메라 좌표계).

| 슬라이스 | 1-based 필드 | 의미 | 변수 |
|----------|--------------|------|------|
| `[26:29]` | 27–29 | 3D gaze target (화면 위 응시점) | `target` |
| `[29:32]` | 30–32 | head rotation (Rodrigues 벡터) | `headrotvectors` |
| `[35:38]` | 36–38 | 오른눈 3D 중심 | `rightcenter` |
| `[38:41]` | 39–41 | 왼눈 3D 중심 | `leftcenter` |

목록의 `left`/`right`에 따라 둘 중 하나의 눈 중심을 정규화 기준점 `c`로 쓴다.

---

## 3. 시점 정규화 (`mpii_norm.Norm`)

목표는 "가상 카메라가 눈 중심을 정면으로, 일정한 거리에서, 머리의 가로축과 수평이 되게 바라보는" 이미지를 만드는 것이다. 이렇게 하면 카메라 위치·거리·머리 roll의 차이가 사라져 모델이 다뤄야 할 변동이 줄어든다.

### 3.1 입력

| 기호 | 코드 | 값 |
|------|------|----|
| `c` | `center` | 눈 중심 (3,) |
| `t` | `gazetarget` | 응시점 (3,) |
| `H` | `headrotvec` | head 회전. (3,)이면 `cv2.Rodrigues`로 3×3로 바꾼다 |
| `K` | `camparams` | 원본 카메라 내부 행렬 (`Camera.mat`의 `cameraMatrix`) |
| `(w, h)` | `imsize` | `(60, 36)` |
| `f_n` | `newfocal` | 960 |
| `d_n` | `newdistance` | 600 (mm) |

### 3.2 회전 R — 가상 카메라의 방향

`d = ‖c‖`라 할 때

```
z = c / d                       # 카메라 → 눈 중심 방향
x_h = H[:, 0]                   # 머리 좌표계의 x축
y = normalize(z × x_h)
x = normalize(y × z)
R = [x; y; z]                   # 행 벡터로 쌓은 3×3
```

- 새 카메라의 z축이 눈 중심을 정확히 향한다 → 눈이 이미지 중앙에 온다.
- x축은 머리의 x축을 z에 수직인 평면으로 투영한 것 → 머리의 roll이 제거된다.

### 3.3 스케일 S — 거리 맞추기

```
S = diag(1, 1, d_n / d)
```

눈이 카메라에서 얼마나 떨어져 있든 600mm 거리에서 찍은 것처럼 크기를 맞춘다.

### 3.4 이미지 warp

```
C_n = [[f_n, 0,   w/2],
       [0,   f_n, h/2],
       [0,   0,   1  ]]          # 가상 카메라 내부 행렬, 주점은 이미지 중앙 (30, 18)

M = S · R                        # 원본 카메라 좌표 → 정규화 좌표
W = C_n · M · K⁻¹                # 원본 픽셀 → 정규화 픽셀 (homography)

eye = cv2.warpPerspective(frame, W, (60, 36))
```

`warpPerspective`의 크기 인자는 `(width, height)`이므로 결과 배열은 **높이 36 × 너비 60**이다. 원본 프레임 전체를 warp하지만 출력 창이 60×36이라 눈 주변만 잘려 나온다.

### 3.5 라벨 변환

| 메서드 | 계산 | 결과 |
|--------|------|------|
| `GetGaze(scale=True)` | `g = t − c`, `normalize(M · g)` | 정규화 좌표계의 3D gaze 단위 벡터 |
| `GetGaze(scale=False)` | `normalize(R · g)` | 회전만 적용한 gaze (`--no-scale`) |
| `GetHeadRot()` | `Rodrigues(M · H)` | 정규화 좌표계의 head 회전 벡터 |
| `GetCoordinate(c)` | `M · c` | 정규화 좌표계의 눈 중심 (`GazeOrigin`) |
| `GetParams()` | `Rodrigues(R)`, `diag(S)` | 라벨에 함께 저장하는 `Rmat`, `Smat` |

참고:

- 기본값 `scale=True`는 GazeHub 기본 동작을 따른 것이다. S는 z 성분만 늘이거나 줄이므로 gaze 방향이 약간 바뀐다. [Zhang et al. 2018](https://doi.org/10.1145/3204493.3204548) ("Revisiting Data Normalization")은 gaze에는 회전만 적용하는 쪽을 권장하며, `--no-scale`이 그 방식이다. 바꾸면 GazeHub가 배포하던 결과와 라벨 값이 달라진다.
- `M · H`는 S 때문에 엄밀한 회전 행렬이 아니다. `cv2.Rodrigues`가 행렬을 회전 벡터로 바꿀 때 가장 가까운 회전으로 맞춰 처리한다.
- `R · c = (0, 0, d)`이므로 `M · c`는 이론상 항상 `(0, 0, 600)`이다. `GazeOrigin` 칸은 형식을 맞추기 위해 남겨 둔 값이다.

---

## 4. 후처리

### 4.1 Histogram equalization

warp 직후 `cv2.equalizeHist`로 밝기 분포를 펴서 조명 차이를 줄인다.

### 4.2 오른눈 → 왼눈 공간으로 반전

모델이 한 종류의 눈만 보도록 오른눈 샘플을 좌우 반전한다 (Zhang et al. 관례).

| 대상 | 처리 |
|------|------|
| 이미지 | `cv2.flip(eye, 1)` (좌우) |
| gaze | `gaze_flip`: x 성분 부호 반전 |
| head | `head_flip`: 회전 행렬의 z축(3번째 열)에서 x 성분 부호를 반전한 뒤 Rodrigues로 다시 벡터화 (GazeHub `HeadFlip`과 동일) |
| origin | x 성분 부호 반전 |

왼눈 샘플은 그대로 둔다. 라벨의 `WhichEye` 칸에는 원래 어느 눈이었는지가 남는다.

### 4.3 3D → 2D

```
gaze_to_2d:  yaw   = atan2(−g_x, −g_z)
             pitch = asin(−g_y)

head_to_2d:  v     = Rodrigues(head)[:, 2]     # 머리 좌표계 z축
             yaw   = atan2(v_x, v_z)
             pitch = asin(v_y)
```

역변환 `gaze_to_3d`는 `g = (−cos(pitch)·sin(yaw), −sin(pitch), −cos(pitch)·cos(yaw))`이다. 라벨에는 2D만 저장하므로 3D 벡터가 필요할 때 이 함수를 쓴다.

단위는 라디안이다. gaze는 카메라를 향하는(−z) 방향이 `(0, 0)`이 되도록 부호를 잡는다. 학습에서는 이 2D 값을 쓴다.

---

## 5. 출력

```
data/EyeBased/MPIIGaze/
  Image/p00/1.jpg … 3000.jpg
  ...
  Label/p00.label … p14.label
```

이미지 파일 이름은 목록 순서대로 붙인 일련번호(1부터)다. 원본 위치는 라벨의 `Origin` 칸에 남는다.

`.label`은 첫 줄이 헤더이고, 이후 한 샘플당 한 줄, 공백으로 구분한다. 벡터는 쉼표로 이어 쓴다.

| 인덱스 | 헤더 | 내용 | 예 |
|--------|------|------|----|
| 0 | `Image` | `Image/` 기준 상대 경로 | `p00/1.jpg` |
| 1 | `Origin` | 원본 프레임 | `day01/0005.jpg` |
| 2 | `WhichEye` | 원래 눈 | `left` / `right` |
| 3 | `3DHead` | 정규화 head 회전 벡터 | `hx,hy,hz` |
| 4 | `2DGaze` | gaze (yaw, pitch) rad | `yaw,pitch` |
| 5 | `2DHead` | head (yaw, pitch) rad | `yaw,pitch` |
| 6 | `Rmat` | R의 Rodrigues 벡터 | `rx,ry,rz` |
| 7 | `Smat` | S의 대각 성분 | `1,1,600/d` |
| 8 | `GazeOrigin` | 정규화 좌표계의 눈 중심 | `ox,oy,oz` |

3D gaze 단위 벡터(`3DGaze`)는 `2DGaze`와 같은 정보라 저장하지 않는다. 필요하면 `mpii_norm.gaze_to_3d`로 계산한다. 예전 라벨에는 `WhichEye`와 `3DHead` 사이에 `3DGaze` 칸이 있었다.

`reader_mpii`는 헤더 이름으로 `Image`(이미지), `2DGaze`(gaze 정답), `2DHead`(head pose 입력)만 찾아 읽으므로 예전 라벨도 그대로 읽는다. 나머지는 다른 방법이나 역변환(정규화 → 원본 카메라 좌표)에 쓸 수 있도록 남긴다.

변환이 끝나면 `_verify_output`이 첫 사람의 `1.jpg`를 읽어 36×60인지 확인하고, 아니면 에러로 멈춘다.

---

## 참고 문헌

- Sugano, Matsushita, Sato. [*Learning-by-Synthesis for Appearance-based 3D Gaze Estimation.*](https://openaccess.thecvf.com/content_cvpr_2014/html/Sugano_Learning-by-Synthesis_for_Appearance-based_2014_CVPR_paper.html) CVPR 2014.
- Zhang, Sugano, Fritz, Bulling. [*Appearance-Based Gaze Estimation in the Wild.*](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.html) CVPR 2015.
- Zhang, Sugano, Bulling. [*Revisiting Data Normalization for Appearance-Based Gaze Estimation.*](https://doi.org/10.1145/3204493.3204548) ETRA 2018.
- Cheng et al. [*Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark.*](https://arxiv.org/abs/2104.12668) (GazeHub)
