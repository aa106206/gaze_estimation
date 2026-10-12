"""leave-one-person-out으로 평가하는 선형 회귀(ridge) baseline. numpy만 쓴다.

입력: 눈 이미지 픽셀 36×60 = 2160개 + 2DHead (yaw, pitch) → 출력: 2DGaze (yaw, pitch)

사람마다 충분 통계량(XᵀX, Xᵀy, 합, 개수)을 한 번만 구해 두고, fold마다 테스트하는 사람을 뺀
14명의 통계량을 더해 정규 방정식을 푼다. 평균·표준편차도 이 학습 통계량에서만 계산하므로
테스트하는 사람의 데이터는 학습에 섞이지 않는다.

    python baseline_linear.py
    python baseline_linear.py --alpha 10 --no-head
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import add

import numpy as np

import reader
from preprocess.mpii_norm import gaze_to_3d

PERSONS = [f"p{k:02d}" for k in range(15)]


def angular_error(pred, gt):
  """pred, gt: (N, 2) (yaw, pitch) 라디안 → (N,) 도"""
  a, b = gaze_to_3d(pred), gaze_to_3d(gt)
  cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
  return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def load_person(root, person, use_head=True, num_workers=8):
  """한 사람의 특징 X (N, D)와 정답 y (N, 2)를 float64로 돌려준다."""
  dataset = reader.loader(os.path.join(root, "Label", f"{person}.label"), os.path.join(root, "Image"))
  with ThreadPoolExecutor(num_workers) as pool:
    samples = list(pool.map(dataset.__getitem__, range(len(dataset))))
  info, y = reader.collate(samples)
  feats = [info["eye"].reshape(len(y), -1)]
  if use_head:
    feats.append(info["head_pose"])
  return np.concatenate(feats, axis=1).astype(np.float64), y.astype(np.float64)


class Stats:
  """정규 방정식에 필요한 충분 통계량. 더하기로 여러 사람을 합친다."""

  def __init__(self, X=None, y=None):
    if X is None:
      return
    self.n = len(X)
    self.sx, self.sy = X.sum(0), y.sum(0)
    self.xx, self.xy = X.T @ X, X.T @ y

  def __add__(self, other):
    s = Stats()
    s.n = self.n + other.n
    s.sx, s.sy = self.sx + other.sx, self.sy + other.sy
    s.xx, s.xy = self.xx + other.xx, self.xy + other.xy
    return s


def fit_ridge(s, alpha):
  """표준화한 특징에 대한 ridge 회귀. 절편은 규제하지 않는다. 원래 단위의 (W, b)를 돌려준다."""
  mx, my = s.sx / s.n, s.sy / s.n
  cxx = s.xx - s.n * np.outer(mx, mx)          # 중심화한 XᵀX
  cxy = s.xy - s.n * np.outer(mx, my)          # 중심화한 Xᵀy
  std = np.sqrt(np.maximum(np.diag(cxx) / s.n, 1e-12))
  A = cxx / np.outer(std, std) + alpha * np.eye(len(std))
  W = np.linalg.solve(A, cxy / std[:, None]) / std[:, None]
  b = my - mx @ W
  return W, b


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--root", default="data/EyeBased/MPIIGaze")
  p.add_argument("--alpha", type=float, default=1000.0, help="ridge 규제 세기 (모든 fold에 같은 값)")
  p.add_argument("--no-head", action="store_true", help="2DHead를 빼고 이미지만 쓴다 (비교용)")
  args = p.parse_args()

  data, stats = {}, {}
  for person in PERSONS:
    X, y = load_person(args.root, person, use_head=not args.no_head)
    data[person] = (X, y)
    stats[person] = Stats(X, y)
  print(f"loaded {sum(len(y) for _, y in data.values())} samples, {X.shape[1]} features "
        f"({'image' if args.no_head else 'image + 2DHead'}), alpha={args.alpha}")

  errors = {}
  for test in PERSONS:
    train = reduce(add, (stats[q] for q in PERSONS if q != test))
    W, b = fit_ridge(train, args.alpha)
    X, y = data[test]
    errors[test] = angular_error(X @ W + b, y).mean()
    print(f"  test {test}: {errors[test]:.2f}°")

  print(f"mean angular error: {np.mean(list(errors.values())):.2f}°")


if __name__ == "__main__":
  main()
