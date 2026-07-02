from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import os
from werkzeug.utils import secure_filename

from search.hybrid_search import search_hybrid
from search.image_search import search_by_image
from config import IMAGE_DIR

app = Flask(__name__)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


def make_result(row, score_col="hybrid_score"):
    image_name = Path(row["image_path"]).name

    return {
        "image_url": f"/images/{image_name}",
        "class_name": row.get("class_name", ""),
        "caption": row.get("caption", ""),
        "viewpoint": row.get("viewpoint", ""),
        "environment": row.get("environment", ""),
        "visible_parts": row.get("visible_parts", ""),
        "score": row.get(score_col, 0),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    mode = "text"

    if request.method == "POST":
        mode = request.form.get("mode", "text")

        if mode == "text":
            query = request.form.get("query", "")

            if query.strip():
                df = search_hybrid(query, top_k=20)
                results = [make_result(row, "hybrid_score") for _, row in df.iterrows()]

        elif mode == "image":
            file = request.files.get("image")

            if file and file.filename:
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_DIR, filename)
                file.save(save_path)

                df = search_by_image(save_path, top_k=20)
                results = [make_result(row, "image_score") for _, row in df.iterrows()]

    return render_template(
        "index.html",
        query=query,
        results=results,
        mode=mode
    )


if __name__ == "__main__":
    app.run(debug=True)