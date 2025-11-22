"""
Automated video production backend, this is what i used so far in this project
- FastAPI
- Django
- Celery
- PostgreSQL
- MinIO S3 storage
- Scene detection (PySceneDetect API)
- Optional CUDA accel (ffmpeg hwaccel / nvenc)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional, Dict, Any, List, Generator, Tuple

from sqlalchemy import create_engine, Column, String, JSON, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from celery import Celery

import boto3, subprocess, datetime, os, json, logging

# LOGGING
logging.basicConfig(
    filename="/tmp/opus.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("opus")


### CONFIG 
REDIS_BROKER = os.environ.get("REDIS_BROKER", "redis://redis:6379/0")
RESULT_BACKEND = os.environ.get("RESULT_BACKEND", REDIS_BROKER)

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "admin123")
S3_BUCKET = os.environ.get("S3_BUCKET", "opus")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./opus.db")
###

celery_app = Celery("opus", broker=REDIS_BROKER, backend=RESULT_BACKEND)

Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Asset(Base):
    __tablename__ = "assets"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String)
    s3_key = Column(String)
    duration = Column(Float, default=0.0)
    meta_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Clip(Base):
    __tablename__ = "clips"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    asset_id = Column(String)
    start_sec = Column(Float)
    end_sec = Column(Float)
    score = Column(Float, default=0.0)
    thumbnails = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

# Ensure bucket exists
try:
    s3.head_bucket(Bucket=S3_BUCKET)
except Exception:
    try:
        s3.create_bucket(Bucket=S3_BUCKET)
        log.info(f"[s3] created bucket {S3_BUCKET}")
    except Exception as e:
        log.warning(f"[s3] create bucket failed: {e}")

def ensure_public_read_bucket():
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "PublicReadAllObjects",
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{S3_BUCKET}/*"]
        }]
    }
    try:
        s3.put_bucket_policy(Bucket=S3_BUCKET, Policy=json.dumps(policy))
        log.info(f"[s3] set public-read policy for {S3_BUCKET}")
    except Exception as e:
        log.warning(f"[s3] public policy not set (ok): {e}")

ensure_public_read_bucket()


def upload_file_to_s3(local_path: str, key: str) -> str:
    s3.upload_file(local_path, S3_BUCKET, key)
    return key

def download_s3_to_local(key: str, local_path: str) -> str:
    s3.download_file(S3_BUCKET, key, local_path)
    return local_path

def has_cuda() -> bool:
    return os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl")

def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ])
        return float(out.decode().strip())
    except Exception:
        return 0.0

def split_evenly(duration: float, num: int, min_len: float) -> List[Tuple[float, float]]:
    """Fallback segments if no scenes."""
    if duration <= 0 or num <= 0:
        return []
    seg_len = max(min_len, duration / num)
    segs = []
    t = 0.0
    for _ in range(num):
        start = t
        end = min(duration, t + seg_len)
        if end - start >= 1.0:
            segs.append((start, end))
        t = end
        if t >= duration:
            break
    return segs


app = FastAPI(title="Opus Starter API")


class CreateProject(BaseModel):
    name: str
    settings: Optional[Dict[str, Any]] = {}


@app.post("/projects")
def create_project(payload: CreateProject, db: Session = Depends(get_db)):
    p = Project(name=payload.name, settings=payload.settings or {})
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"project_id": p.id}


@app.post("/projects/{project_id}/assets")
async def upload_asset(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    local_file = f"/tmp/{uuid4().hex}_{file.filename}"
    with open(local_file, "wb") as f:
        f.write(await file.read())

    s3_key = f"assets/{uuid4().hex}_{file.filename}"
    upload_file_to_s3(local_file, s3_key)

    asset = Asset(
        project_id=project_id,
        s3_key=s3_key,
        meta_data={"filename": file.filename},
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    ingest_asset.delay(asset.id)
    return {"asset_id": asset.id}


@app.get("/projects/{project_id}/clips")
def list_clips(project_id: str, db: Session = Depends(get_db)):
    assets = db.query(Asset).filter(Asset.project_id == project_id).all()
    response: List[Dict[str, Any]] = []
    for a in assets:
        clips = db.query(Clip).filter(Clip.asset_id == a.id).all()
        for c in clips:
            response.append({
                "clip_id": c.id,
                "asset_id": c.asset_id,
                "start": c.start_sec,
                "end": c.end_sec,
                "score": c.score,
                "thumbnails": c.thumbnails,
            })
    return response


@app.post("/projects/{project_id}/clips/{clip_id}/generate")
def generate_clip(project_id: str, clip_id: str):
    job = generate_pipeline.delay(clip_id, {})
    return {"job_id": job.id}


@app.get("/outputs/{clip_id}")
def get_output(clip_id: str):
    key = f"outputs/{clip_id}_final.mp4"
    local_path = f"/tmp/{clip_id}_final.mp4"
    try:
        download_s3_to_local(key, local_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Output not found")

    return FileResponse(local_path, media_type="video/mp4")


@app.get("/projects/{project_id}/status")
def project_status(project_id: str, db: Session = Depends(get_db)):
    assets = db.query(Asset).filter(Asset.project_id == project_id).all()
    asset_ids = [a.id for a in assets]

    clips = []
    if asset_ids:
        clips = db.query(Clip).filter(Clip.asset_id.in_(asset_ids)).all()

    clip_count = len(clips)
    thumbs_ready = sum(1 for c in clips if isinstance(c.thumbnails, dict) and c.thumbnails.get("thumb_s3"))

    finals_ready = 0
    for c in clips:
        key = f"outputs/{c.id}_final.mp4"
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=key)
            finals_ready += 1
        except Exception:
            pass

    return {
        "clip_count": clip_count,
        "thumbs_ready": thumbs_ready,
        "finals_ready": finals_ready
    }


@app.get("/projects/{project_id}/clips/json", include_in_schema=False)
@app.get("/projects/{project_id}/clips/json/", include_in_schema=False)
def clips_json(project_id: str, db: Session = Depends(get_db)):
    assets = db.query(Asset).filter(Asset.project_id == project_id).all()
    out = []
    for a in assets:
        clips = db.query(Clip).filter(Clip.asset_id == a.id).all()
        for c in clips:
            thumb_key = None
            if isinstance(c.thumbnails, dict):
                thumb_key = c.thumbnails.get("thumb_s3")
            out.append({
                "clip_id": c.id,
                "start": c.start_sec,
                "end": c.end_sec,
                "duration": float(c.end_sec - c.start_sec),
                "thumb": thumb_key,
                "output_url": f"/outputs/{c.id}",
            })
    return out


@app.get("/logs")
def get_logs():
    log_path = "/tmp/opus.log"
    if not os.path.exists(log_path):
        return {"logs": "No logs yet."}
    with open(log_path, "r") as f:
        return {"logs": f.read()}


# ================================================================
# CELERY TASKS
# ================================================================
@celery_app.task(bind=True)
def ingest_asset(self, asset_id: str):
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        if not asset:
            return {"error": "asset not found"}

        local = f"/tmp/{asset.id}.mp4"
        download_s3_to_local(asset.s3_key, local)

        dur = ffprobe_duration(local)
        asset.duration = dur
        db.add(asset)
        db.commit()

        log.info(f"[ingest_asset] asset={asset.id} duration={dur:.2f}s")

        # load project settings
        project = db.query(Project).filter(Project.id == asset.project_id).first()
        settings = (project.settings or {}) if project else {}
        num_clips = int(settings.get("num_clips", 2))
        min_clip_sec = float(settings.get("min_clip_sec", 30))

        clip_detect.delay(asset.id, num_clips, min_clip_sec)

        return {"asset_id": asset.id, "duration": dur}
    finally:
        db.close()


@celery_app.task(bind=True)
def clip_detect(self, asset_id: str, number_of_clips: int = 2, min_clip_sec: float = 30.0):
    from scenedetect import detect, ContentDetector

    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        if not asset:
            return {"error": "asset not found"}

        local = f"/tmp/{asset.id}.mp4"
        download_s3_to_local(asset.s3_key, local)

        log.info(f"[clip_detect] asset={asset.id} num={number_of_clips} min_len={min_clip_sec}s")

        scenes = detect(local, ContentDetector(threshold=27.0))
        segments = [(s.get_seconds(), e.get_seconds()) for (s, e) in scenes]

        duration = asset.duration or ffprobe_duration(local)

        # Filter tiny scenes + normalize length to min_clip_sec
        filtered = [(st, en) for (st, en) in segments if en - st >= 2.0]
        segments = filtered

        # If no scenes detected, fallback to even split
        if not segments:
            segments = split_evenly(duration, number_of_clips, min_clip_sec)

        # If too many segments, downsample evenly
        if len(segments) > number_of_clips:
            step = max(1, len(segments) // number_of_clips)
            segments = [segments[i] for i in range(0, len(segments), step)][:number_of_clips]

        created = 0
        TARGET_CLIP_LENGTH = 30.0
        for start, end in segments:
            forced_end = start + TARGET_CLIP_LENGTH

            # ensure we don't go past the video duration
            if forced_end > asset.duration:
                forced_end = asset.duration

            if forced_end - start < 2:
                continue

            clip = Clip(asset_id=asset.id, start_sec=start, end_sec=end, score=1.0)
            db.add(clip)
            db.commit()
            db.refresh(clip)

            # thumbnail (single frame)
            thumb = f"/tmp/{clip.id}.webp"
            subprocess.call([
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", local,
                "-vf", "scale=320:-1",
                "-frames:v", "1",
                "-update", "1",
                thumb,
            ])

            thumb_key = f"thumbnails/{clip.id}.webp"
            upload_file_to_s3(thumb, thumb_key)

            clip.thumbnails = {"thumb_s3": thumb_key}
            db.add(clip)
            db.commit()

            created += 1
            log.info(f"[clip_detect] created clip={clip.id} {start:.2f}->{end:.2f} thumb={thumb_key}")

        return {"clips_created": created}
    finally:
        db.close()


def asr_worker(path: str):
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(path)
        return result.get("text", "").strip(), []
    except Exception as e:
        log.warning(f"[asr] whisper failed: {e}")
        return "", []


def tts_worker(text: str, voice: str = "default"):
    out = f"/tmp/tts_{uuid4().hex}.wav"
    subprocess.call([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "2",
        out
    ])
    return out


def extract_clip_with_audio(src: str, start: float, end: float, out_path: str):
    """Extract clip portion WITH audio"""
    use_cuda = has_cuda()
    vcodec = "h264_nvenc" if use_cuda else "libx264"

    if use_cuda:
        log.info("[extract] CUDA_ON=True (nvenc)")
    else:
        log.info("[extract] CUDA_ON=False")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", src,
        "-c:v", vcodec,
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ]
    subprocess.call(cmd)


def render_worker(clip_path: str, voice_path: str):
    """Mix original clip audio with voice (if any), output mp4."""
    out = f"/tmp/final_{uuid4().hex}.mp4"

    use_cuda = has_cuda()
    vcodec = "h264_nvenc" if use_cuda else "libx264"
    if use_cuda:
        log.info("[render] CUDA_ON is True (nvenc)")
    else:
        log.info("[render] CUDA_ON is False")

    # amix to keep original audio + voice
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-i", voice_path,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest",
        "-c:v", vcodec,
        "-preset", "fast",
        "-c:a", "aac",
        "-shortest",
        out
    ]
    subprocess.call(cmd)
    return out


@celery_app.task(bind=True)
def generate_pipeline(self, clip_id: str, opts: dict):
    """ASR -> TTS -> Render -> Upload final output."""
    db = SessionLocal()
    try:
        log.info(f"[generate_pipeline] START clip={clip_id}")

        clip = db.query(Clip).filter(Clip.id == clip_id).first()
        if not clip:
            log.error("[generate_pipeline] clip not found")
            return {"error": "clip not found"}

        asset = db.query(Asset).filter(Asset.id == clip.asset_id).first()
        if not asset:
            log.error("[generate_pipeline] asset not found")
            return {"error": "asset not found"}

        local_asset = f"/tmp/{asset.id}.mp4"
        download_s3_to_local(asset.s3_key, local_asset)

        if not os.path.exists(local_asset):
            log.error("[generate_pipeline] asset missing on disk")
            return {"error": "asset missing"}

        # Extract clip (with audio)
        clip_file = f"/tmp/{clip.id}.mp4"
        extract_clip_with_audio(local_asset, clip.start_sec, clip.end_sec, clip_file)

        if not os.path.exists(clip_file):
            log.error("[generate_pipeline] clip extract failed")
            return {"error": "clip extract failed"}

        transcript, word_ts = asr_worker(clip_file)
        voice_name = (opts or {}).get("voice", "default")
        voice_file = tts_worker(transcript, voice_name)

        final_file = render_worker(clip_file, voice_file)

        if not os.path.exists(final_file):
            log.error("[generate_pipeline] render failed")
            return {"error": "render failed"}

        final_key = f"outputs/{clip_id}_final.mp4"
        upload_file_to_s3(final_file, final_key)

        log.info(f"[generate_pipeline] DONE clip={clip_id} key={final_key}")
        return {"final_s3_key": final_key}
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("opus_starter_bot:app", host="0.0.0.0", port=8000, reload=True)
