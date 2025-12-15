from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_pdf_report(out_path: Path, summary: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=letter)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "NeuroAid AI Screening Report (Informational)")

    c.setFont("Helvetica", 10)
    c.drawString(50, 730, "This report is not a diagnosis. Use it to guide next steps with educators/specialists.")

    y = 700
    c.setFont("Helvetica", 11)
    for k, v in summary.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18
        if y < 60:
            c.showPage()
            y = 750

    c.save()
    return out_path
