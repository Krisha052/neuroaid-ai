import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.config import CONFIG
from src.utils.pdf_report import export_pdf_report

from .errors import ScreeningError
from .service import load_prompts, resolve_prompt_text, run_screening

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroaid.api")

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_mb * 1024 * 1024

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": CONFIG.model_path.exists()})

    @app.get("/api/v1/prompts")
    def prompts():
        sections = load_prompts(CONFIG.sample_prompts_path)
        return jsonify({name: " ".join(lines) for name, lines in sections.items()})

    @app.post("/api/v1/screen")
    def screen():
        if "file" not in request.files:
            return _error_response(
                ScreeningError("No 'file' field in multipart form data."), 400
            )
        file = request.files["file"]
        if file.filename == "":
            return _error_response(ScreeningError("Empty filename."), 400)

        prompt_set = request.form.get("prompt_set")
        prompt_text = request.form.get("prompt_text")
        mode = request.form.get("transcription_mode", "local")

        resolved_prompt = resolve_prompt_text(prompt_set, prompt_text)
        result = run_screening(file.read(), resolved_prompt, transcription_mode=mode)
        return jsonify(result)

    @app.post("/api/v1/screen/report")
    def screen_report():
        if "file" not in request.files:
            return _error_response(
                ScreeningError("No 'file' field in multipart form data."), 400
            )
        file = request.files["file"]
        prompt_set = request.form.get("prompt_set")
        prompt_text = request.form.get("prompt_text")
        mode = request.form.get("transcription_mode", "local")

        resolved_prompt = resolve_prompt_text(prompt_set, prompt_text)
        result = run_screening(file.read(), resolved_prompt, transcription_mode=mode)

        CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = CONFIG.reports_dir / "neuroaid_report.pdf"
        summary = {**result["features"], **(result.get("risk_assessment") or {})}
        export_pdf_report(pdf_path, summary)

        from flask import send_file
        return send_file(pdf_path, mimetype="application/pdf",
                          as_attachment=True, download_name="neuroaid_report.pdf")

    @app.errorhandler(ScreeningError)
    def handle_screening_error(err: ScreeningError):
        return _error_response(err, err.status_code)

    @app.errorhandler(413)
    def handle_too_large(_err):
        return jsonify({"error": f"Upload exceeds {CONFIG.max_upload_mb}MB limit.",
                         "type": "PayloadTooLargeError"}), 413

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        if isinstance(err, HTTPException):
            return jsonify({"error": err.description, "type": err.name}), err.code
        logger.exception("Unhandled error in API request")
        return jsonify({"error": "Internal server error.", "type": "InternalError"}), 500

    return app

def _error_response(err: ScreeningError, status_code: int):
    return jsonify({"error": err.message, "type": err.__class__.__name__}), status_code
