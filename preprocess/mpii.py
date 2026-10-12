"""MPIIGaze 다운로드와 GazeHub eye-based 전처리.

원본 데이터셋(비영리 연구 목적)을 받은 뒤, GazeHub와 같은 눈 이미지(36×60 grayscale)와
`.label` 파일을 만든다. `reader_mpii`가 이 출력을 그대로 읽는다.

사용 예:

    python -m preprocess.mpii all
    python -m preprocess.mpii download
    python -m preprocess.mpii convert --dataset data/MPIIGaze --out-root data/EyeBased/MPIIGaze
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
SAMPLE_DIR = Path("Evaluation Subset") / "sample list for eye image"
ORIGINAL_DIR = Path("Data") / "Original"
IMAGE_SIZE = (60, 36)  # warpPerspective (width, height) → 저장된 배열은 36×60
# 3DGaze는 2DGaze와 같은 정보라 저장하지 않는다. 필요하면 mpii_norm.gaze_to_3d로 복원한다.
HEADER = "Image Origin WhichEye 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin"

cv2 = np = sio = dpc = None


def _deps():
    global cv2, np, sio, dpc
    if cv2 is not None:
        return
    try:
        import cv2 as _cv2
        import numpy as _np
        import scipy.io as _sio
        from preprocess import mpii_norm as _dpc
    except ImportError as exc:
        raise SystemExit(
            "전처리에는 numpy, opencv-python, scipy가 필요하다.\n"
            "  pip install numpy opencv-python scipy"
        ) from exc
    cv2, np, sio, dpc = _cv2, _np, _sio, _dpc


def find_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path, path / "MPIIGaze", path / "Origin", path / "MPIIGaze" / "Origin"]
    for root in candidates:
        if (root / ORIGINAL_DIR).is_dir() and (root / SAMPLE_DIR).is_dir():
            return root
    raise FileNotFoundError(
        f"MPIIGaze 원본을 찾지 못했다: {path}\n"
        f"  필요한 경로: <root>/{ORIGINAL_DIR} 와 <root>/{SAMPLE_DIR}\n"
        "  먼저 `python -m preprocess.mpii download` 를 실행하거나 --dataset 을 지정하라."
    )


def _fmt(vec) -> str:
    return ",".join(np.asarray(vec).astype(str).tolist())


def decode_annotation(line: str) -> dict:
    """MPIIGaze Original annotation.txt 한 줄.

    dim 1–24: 랜드마크, 27–29: 3D gaze target, 30–35: head pose,
    36–38: 오른눈 중심, 39–41: 왼눈 중심. (1-based 설명, 코드는 0-based)
    """
    annotation = np.array(line.strip().split(), dtype=np.float64)
    return {
        "headrotvectors": annotation[29:32],
        "rightcenter": annotation[35:38],
        "leftcenter": annotation[38:41],
        "target": annotation[26:29],
    }


def load_day_annotations(person_dir: Path) -> dict[str, list[str]]:
    anno = {}
    for day_dir in sorted(person_dir.iterdir()):
        if not day_dir.is_dir() or day_dir.name in {"Calibration"} or day_dir.name.startswith("."):
            continue
        txt = day_dir / "annotation.txt"
        if not txt.is_file():
            raise FileNotFoundError(txt)
        with open(txt) as f:
            anno[day_dir.name] = f.readlines()
    return anno


def convert_person(
    person: str,
    person_dir: Path,
    sample_list: Path,
    image_out: Path,
    label_out: Path,
    scale: bool = True,
) -> int:
    camera = np.asarray(sio.loadmat(str(person_dir / "Calibration" / "Camera.mat"))["cameraMatrix"]).reshape(3, 3)
    anno_dict = load_day_annotations(person_dir)
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)

    with open(sample_list) as f:
        im_list = [ln for ln in f.readlines() if ln.strip()]

    written = 0
    with open(label_out, "w") as outfile:
        outfile.write(HEADER + "\n")
        total = len(im_list)
        for count, info in enumerate(im_list):
            im_info, which_eye = info.strip().split()
            day, im_name = im_info.split("/")
            im_number = int(Path(im_name).stem)

            im_path = person_dir / day / im_name
            im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"이미지를 읽지 못했다: {im_path}")

            annotation = decode_annotation(anno_dict[day][im_number - 1])
            eye_key = "leftcenter" if which_eye == "left" else "rightcenter"
            norm = dpc.Norm(
                center=annotation[eye_key],
                gazetarget=annotation["target"],
                headrotvec=annotation["headrotvectors"],
                imsize=IMAGE_SIZE,
                camparams=camera,
            )
            origin = norm.GetCoordinate(annotation[eye_key])
            im_eye = cv2.equalizeHist(norm.GetImage(im))
            gaze = norm.GetGaze(scale=scale)
            head = norm.GetHeadRot(vector=True)

            if which_eye != "left":
                im_eye = cv2.flip(im_eye, 1)
                gaze = dpc.gaze_flip(gaze)
                head = dpc.head_flip(head)
                origin = origin.copy()
                origin[0] = -origin[0]

            gaze_2d = dpc.gaze_to_2d(gaze)
            head_2d = dpc.head_to_2d(head)
            rvec, svec = norm.GetParams()

            save_name = f"{count + 1}.jpg"
            cv2.imwrite(str(image_out / save_name), im_eye)

            rel_image = f"{person}/{save_name}"
            row = " ".join(
                [
                    rel_image,
                    im_info,
                    which_eye,
                    _fmt(head),
                    _fmt(gaze_2d),
                    _fmt(head_2d),
                    _fmt(rvec),
                    _fmt(svec),
                    _fmt(origin),
                ]
            )
            outfile.write(row + "\n")
            written += 1
            if (count + 1) % 100 == 0 or count + 1 == total:
                print(f"\r  {person}: {count + 1}/{total}", end="", flush=True)
        print()
    return written


def convert(dataset: Path, out_root: Path, persons: list[str] | None = None, scale: bool = True) -> None:
    _deps()
    root = find_dataset_root(dataset)
    sample_root = root / SAMPLE_DIR
    original_root = root / ORIGINAL_DIR
    out_root = out_root.expanduser().resolve()

    sample_files = sorted(p for p in sample_root.iterdir() if p.suffix == ".txt" and p.stem.startswith("p"))
    if persons:
        want = set(persons)
        sample_files = [p for p in sample_files if p.stem in want]
        missing = want - {p.stem for p in sample_files}
        if missing:
            raise FileNotFoundError(f"sample list에 없는 person: {sorted(missing)}")

    if not sample_files:
        raise FileNotFoundError(f"sample list가 비어 있다: {sample_root}")

    print(f"dataset: {root}")
    print(f"output:  {out_root}")
    total_written = 0
    for sample_list in sample_files:
        person = sample_list.stem
        n = convert_person(
            person=person,
            person_dir=original_root / person,
            sample_list=sample_list,
            image_out=out_root / "Image" / person,
            label_out=out_root / "Label" / f"{person}.label",
            scale=scale,
        )
        total_written += n
        print(f"  saved {n} samples → Image/{person}, Label/{person}.label")

    _verify_output(out_root, len(sample_files), total_written)


def _verify_output(out_root: Path, n_person: int, n_sample: int) -> None:
    labels = sorted((out_root / "Label").glob("*.label"))
    probe = out_root / "Image" / labels[0].stem / "1.jpg"
    img = cv2.imread(str(probe), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"전처리 결과를 읽지 못했다: {probe}")
    h, w = img.shape
    print(f"done: {n_person} persons, {n_sample} samples, first image {h}×{w} (expect 36×60)")
    if (h, w) != (36, 60):
        raise RuntimeError(f"눈 이미지 크기가 36×60이 아니다: {h}×{w}")


def _reporthook(block: int, block_size: int, total: int) -> None:
    got = block * block_size
    if total > 0:
        pct = min(100.0, 100.0 * got / total)
        print(f"\r  download {got / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
    else:
        print(f"\r  download {got / 1e6:.1f} MB", end="", flush=True)


def download(url: str, archive: Path, dataset: Path) -> Path:
    archive = archive.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)

    try:
        root = find_dataset_root(dataset)
        print(f"이미 원본이 있다: {root}")
        return root
    except FileNotFoundError:
        pass

    if not archive.is_file():
        print(f"downloading {url}")
        print("MPIIGaze는 비영리 연구 목적으로만 사용할 수 있다.")
        tmp = archive.with_suffix(archive.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, _reporthook)
            print()
            tmp.replace(archive)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    else:
        print(f"archive exists: {archive}")

    extract_to = dataset.parent
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} → {extract_to}")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(extract_to, filter="data")
        except TypeError:
            tar.extractall(extract_to)
    root = find_dataset_root(dataset)
    print(f"extracted: {root}")
    return root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and GazeHub-preprocess MPIIGaze")
    p.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=["download", "convert", "all"],
        help="download tar.gz, convert Original→Image/Label, or both (default: all)",
    )
    p.add_argument("--url", default=DEFAULT_URL, help="MPIIGaze tar.gz URL")
    p.add_argument("--archive", type=Path, default=Path("data/raw/MPIIGaze.tar.gz"))
    p.add_argument("--dataset", type=Path, default=Path("data/MPIIGaze"), help="extracted MPIIGaze root")
    p.add_argument("--out-root", type=Path, default=Path("data/EyeBased/MPIIGaze"))
    p.add_argument("--persons", nargs="*", help="optional subset, e.g. p00 p01")
    p.add_argument("--no-scale", action="store_true", help="use R only (GazeHub scale=False)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "all"
    scale = not args.no_scale

    if cmd in {"download", "all"}:
        download(args.url, args.archive, args.dataset)
        if cmd == "download":
            return 0

    if cmd in {"convert", "all"}:
        convert(args.dataset, args.out_root, persons=args.persons, scale=scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
