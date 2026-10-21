import os
import math
import torch
import gdown
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired, NumberRange
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from Utils.models import VGGEncoder, Decoder
from Utils.utils import adaptive_instance_normalization


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_PATH = os.path.join(BASE_DIR, "vgg_normalised.pth")
CKPT_DIR = os.path.join(BASE_DIR, "experiment", "adain_60-100")
CKPT_PATH = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def download_model(file_id, output_path):
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, "rb") as file:
            header = file.read(512).lstrip().lower()

        if not header.startswith((b"<html", b"<!doctype", b"<head")):
            return

        os.remove(output_path)

    result = gdown.download(
        id=file_id,
        output=output_path,
        quiet=False
    )

    if not result or not os.path.isfile(output_path):
        raise RuntimeError("Model download failed: " + output_path)

    if os.path.getsize(output_path) == 0:
        raise RuntimeError("Downloaded model file is empty: " + output_path)

    with open(output_path, "rb") as file:
        header = file.read(512).lstrip().lower()

    if header.startswith((b"<html", b"<!doctype", b"<head")):
        os.remove(output_path)
        raise RuntimeError(
            "Google Drive returned HTML instead of model weights. "
            "Check the file ID and sharing permissions."
        )


download_model(
    "1Vp155Ft7Ps7f01SCUtTqUu7bBmEFqNXb",
    VGG_PATH
)

download_model(
    "1oHt4zqCDaiFymw5mDHyau725JYhfEABp",
    CKPT_PATH
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)


class UploadForm(FlaskForm):
    content = FileField("Content Image", validators=[InputRequired()])
    style = FileField("Style Image", validators=[InputRequired()])
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[
            InputRequired(),
            NumberRange(min=0.0, max=1.0)
        ]
    )
    submit = SubmitField("Transfer Style")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)

try:
    checkpoint = torch.load(
        CKPT_PATH,
        map_location=device,
        weights_only=True
    )
except Exception as exc:
    raise RuntimeError(
        "Could not load the decoder checkpoint. Verify that the downloaded "
        "file is a valid PyTorch checkpoint and matches your installed "
        "PyTorch version."
    ) from exc

if not isinstance(checkpoint, dict):
    raise RuntimeError("Unexpected checkpoint format: expected a dictionary.")

if "decoder_state_dict" not in checkpoint:
    raise RuntimeError(
        "The checkpoint does not contain 'decoder_state_dict'. "
        "Available keys: " + ", ".join(map(str, checkpoint.keys()))
    )

decoder.load_state_dict(checkpoint["decoder_state_dict"])

print(
    "Model loaded successfully. Epoch:",
    checkpoint.get("epoch", "unknown"),
    "Device:",
    device
)

encoder.eval()
decoder.eval()
encoder.requires_grad_(False)
decoder.requires_grad_(False)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(content_image, style_image, alpha):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    content_tensor = image_transform(content_image).unsqueeze(0).to(device)
    style_tensor = image_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        stylized_features = (
            alpha * stylized_features
            + (1.0 - alpha) * content_features
        )

        result = decoder(stylized_features)

    return result


def save_image(image_tensor, output_path):
    image_tensor = image_tensor.detach().cpu().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image_tensor)
    image.save(output_path, format="PNG")


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        content_upload = form.content.data
        style_upload = form.style.data

        if not content_upload or not content_upload.filename:
            error = "Please upload a content image."
        elif not style_upload or not style_upload.filename:
            error = "Please upload a style image."
        elif not allowed_file(content_upload.filename):
            error = "Content image must be PNG, JPG, or JPEG."
        elif not allowed_file(style_upload.filename):
            error = "Style image must be PNG, JPG, or JPEG."
        else:
            content_filename = secure_filename(content_upload.filename)
            style_filename = secure_filename(style_upload.filename)

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                content_filename
            )
            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                style_filename
            )

            try:
                content_upload.save(content_path)
                style_upload.save(style_path)

                with Image.open(content_path) as image:
                    content_image = image.convert("RGB")

                with Image.open(style_path) as image:
                    style_image = image.convert("RGB")

                alpha = float(form.alpha.data)

                if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                    raise ValueError("Alpha must be between 0 and 1.")

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                result_filename = (
                    "stylized_" + os.path.splitext(content_filename)[0] + ".png"
                )
                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(stylized_image, result_path)
                result_image = result_filename

            except (UnidentifiedImageError, OSError):
                error = "Could not read an uploaded image. Please try another image."
            except Exception as exc:
                app.logger.exception("Style transfer failed")
                error = "Style transfer failed. Please check the server logs."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, "examples"),
        filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
