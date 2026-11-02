import os
import shutil
import subprocess
import cv2


def _find_ffmpeg():
    # 1. Check system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Check bundled imageio_ffmpeg binary
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(exe):
            return exe
    except Exception:
        pass

    # 3. Check Windows WinGet install locations if on Windows
    if os.name == "nt":
        winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
        if os.path.exists(winget_base):
            for root, _, files in os.walk(winget_base):
                if "ffmpeg.exe" in files:
                    return os.path.join(root, "ffmpeg.exe")

    return "ffmpeg"


def read_video(path, max_frames=None, start_frame=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file does not exist: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps != fps:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    count = 0
    while True:
        if max_frames is not None and count >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        count += 1

    cap.release()
    return frames, fps, (w, h)


def write_video(frames, path, fps, size):
    if not frames:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stem, ext = os.path.splitext(path)
    tmp = f"{stem}_raw{ext or '.mp4'}"

    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not out.isOpened():
        raise RuntimeError(f"Could not initialize OpenCV VideoWriter for {tmp}")

    for f in frames:
        out.write(f)
    out.release()

    ffmpeg_bin = _find_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        tmp,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception as e:
        print(f"[warning] ffmpeg encoding fallback triggered ({e}). Using raw mp4.")
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)


if __name__ == "__main__":
    clip = "data/clips/rally_sample.mp4"
    if os.path.exists(clip):
        frames, fps, size = read_video(clip, max_frames=10)
        print(f"Sample test: read {len(frames)} frames, {fps:.1f} FPS, {size}")
