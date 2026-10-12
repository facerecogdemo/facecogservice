import hmac
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from model.face_model import FaceModel


MAX_IMAGE_SIZE = 5 * 1024 * 1024
SERVICE_TOKEN = os.environ["FACE_SERVICE_TOKEN"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.face_model = FaceModel()
    yield


app = FastAPI(
    title="Face Recognition Service",
    version="1.0.0",
    lifespan=lifespan,
)


def verify_token(authorization: str | None):
    expected = f"Bearer {SERVICE_TOKEN}"

    if not authorization or not hmac.compare_digest(
        authorization, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid service credentials.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "face_model"),
    }


@app.post("/embedding")
async def embedding(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    verify_token(authorization)

    if image.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    raw_image = await image.read(MAX_IMAGE_SIZE + 1)

    if not raw_image:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(raw_image) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller.",
        )

    try:
        vector = app.state.face_model.get_embedding(raw_image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "status": "success",
        "embedding": vector,
    }
