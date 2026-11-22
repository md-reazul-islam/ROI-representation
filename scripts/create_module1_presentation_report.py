from __future__ import annotations

from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "Module1_Presentation_Report.docx"


def font(size=24, bold=False):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_pipeline_diagram(path: Path):
    width, height = 1800, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(44, True)
    label = font(25, True)
    small = font(20, False)
    draw.text((55, 30), "Module 1: From Video to Infant-Specific Representation", fill=(25, 35, 45), font=title)
    stages = [
        ("1. Dataset", "manifest + validation", (43, 91, 138)),
        ("2. Detection", "YOLO + RT-DETR", (37, 126, 145)),
        ("3. Fusion", "IoU + confidence", (48, 145, 95)),
        ("4. Tracking", "ByteTrack / BoT-SORT", (114, 90, 160)),
        ("5. Recovery", "temporal prediction", (194, 117, 45)),
        ("6. Pose", "Sapiens2: 308 keypoints", (174, 72, 86)),
        ("7. Anatomy", "dynamic candidate ROIs", (42, 126, 120)),
        ("8. Quality", "visibility + motion + blur", (76, 98, 150)),
        ("9. Selection", "confidence-weighted ROIs", (154, 90, 55)),
        ("10. Evaluation", "B0-B5 ablation", (45, 45, 45)),
    ]
    x_positions = [70, 435, 800, 1165, 1530]
    y_positions = [160, 420]
    box_w, box_h = 290, 155
    for index, (heading, body, color) in enumerate(stages):
        row, col = divmod(index, 5)
        x, y = x_positions[col], y_positions[row]
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=18, fill=color)
        draw.text((x + 18, y + 24), heading, fill="white", font=label)
        draw.multiline_text((x + 18, y + 75), body, fill="white", font=small, spacing=5)
        if col < 4:
            draw.line((x + box_w + 8, y + box_h / 2, x + box_w + 65, y + box_h / 2), fill=(70, 70, 70), width=5)
            draw.polygon([(x + box_w + 65, y + box_h / 2), (x + box_w + 47, y + box_h / 2 - 12), (x + box_w + 47, y + box_h / 2 + 12)], fill=(70, 70, 70))
    draw.line((1675, 315, 1730, 315, 1730, 500, 70, 500), fill=(70, 70, 70), width=5)
    draw.polygon([(70, 500), (88, 488), (88, 512)], fill=(70, 70, 70))
    draw.text((70, 650), "Module 1 ends with an infant-specific visual representation. HR/RR estimation is outside this module.", fill=(35, 35, 35), font=font(24, False))
    image.save(path)


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(document, headers, rows, widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, color=(255, 255, 255), size=9)
        shade(table.rows[0].cells[index], "2F6574")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value, size=8.5)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph()
    return table


def add_caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(80, 80, 80)


def add_image_row(document, images, captions, width=2.15):
    table = document.add_table(rows=1, cols=len(images))
    for index, image in enumerate(images):
        cell = table.rows[0].cells[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if image.exists():
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(image), width=Inches(width))
        caption = cell.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(captions[index])
        run.italic = True
        run.font.size = Pt(8)
    document.add_paragraph()


def add_bullet(document, text, level=0):
    paragraph = document.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run(text)


def build_document():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = FIGURE_DIR / "module1_pipeline_diagram.png"
    make_pipeline_diagram(pipeline)

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(.6)
    section.bottom_margin = Inches(.6)
    section.left_margin = Inches(.7)
    section.right_margin = Inches(.7)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    styles["Heading 1"].font.color.rgb = RGBColor(47, 101, 116)
    styles["Heading 2"].font.color.rgb = RGBColor(47, 101, 116)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Module 1\nInfant-Specific Visual Representation for Physiological Monitoring")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(35, 77, 88)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(f"Presentation report | Prepared {date.today().isoformat()}").italic = True
    document.add_paragraph()
    document.add_picture(str(pipeline), width=Inches(6.8))
    add_caption(document, "Figure 1. Complete Module 1 processing architecture.")

    document.add_heading("Executive Summary", level=1)
    document.add_paragraph(
        "Module 1 converts raw infant video into an anatomically informed, temporally stable, quality-weighted visual representation. "
        "The pipeline begins with independent YOLO and RT-DETR detectors, combines their evidence, maintains infant continuity over time, "
        "uses Sapiens2 pose landmarks to define anatomical regions, and dynamically selects the most reliable regions for later physiological analysis."
    )
    add_bullet(document, "The central research question is whether pose-, anatomy-, visibility-, and motion-aware ROI selection is more reliable than a fixed ROI.")
    add_bullet(document, "Module 1 ends before HR and RR estimation; it produces the visual representation used by later modules.")
    add_bullet(document, "All truth-dependent results are reported only when valid annotations are available.")

    document.add_heading("Phase-by-Phase Methodology", level=1)
    phase_rows = [
        ("0-1", "Dataset contract", "Videos, metadata", "Validation, manifest, reproducible splits", "dataset_manifest.csv"),
        ("2", "YOLO baseline", "Video frames", "Object detection", "YOLO boxes, confidence, metrics"),
        ("3", "RT-DETR baseline", "Same frames as YOLO", "Independent object detection", "RT-DETR boxes and metrics"),
        ("4", "Detection fusion", "YOLO + RT-DETR", "IoU matching, OR/AND/weighted fusion", "Fused infant detections"),
        ("5", "Tracking", "Fused detections", "ByteTrack and BoT-SORT", "Track IDs and continuity metrics"),
        ("6", "Temporal recovery", "Detector + tracker history", "Prediction, velocity, scale, IoU checks", "Recovered boxes and recovery confidence"),
        ("7", "Pose estimation", "Final infant box", "Sapiens2 308 keypoints", "Original-frame keypoints"),
        ("8", "Anatomical ROIs", "Keypoints + box", "Infant-relative anatomical mapping", "Dynamic candidate ROIs"),
        ("9", "ROI quality", "ROIs + frames", "Skin, blur, illumination, motion, temporal features", "Normalized quality features"),
        ("10", "Dynamic selection", "Quality features", "Q score and softmax alpha weights", "Infant-specific representation"),
        ("11", "Ablation", "All outputs + ground truth", "B0-B5 comparison and statistics", "Evaluation reports and figures"),
    ]
    add_table(document, ["Phase", "Module", "Input", "Method", "Output"], phase_rows, [0.45, 1.15, 1.55, 2.35, 1.55])

    document.add_heading("Visual Evidence from the Implemented Pipeline", level=1)
    yolo = ROOT / "outputs/yolo/visualizations/VID_001_frame_00000000.jpg"
    rtdetr = ROOT / "outputs/rtdetr/visualizations/VID_001_frame_00000000.jpg"
    fusion = ROOT / "outputs/fusion/visualizations/VID_001_frame_00000000.jpg"
    recovery = ROOT / "outputs/recovery/visualizations/VID_001_frame_00000000.jpg"
    tracking = ROOT / "outputs/tracking/bytetrack/visualizations/VID_001_frame_00000010.jpg"
    pose = ROOT / "outputs/pose_smoke/visualizations/VID_001_frame_00000000.jpg"
    add_image_row(document, [yolo, rtdetr], ["YOLO detection", "RT-DETR detection"])
    add_caption(document, "Figure 2. Independent detectors process the same infant frame.")
    add_image_row(document, [fusion, recovery, tracking], ["Detector fusion", "Temporal recovery", "Confirmed track"] , width=2.05)
    add_caption(document, "Figure 3. Fusion, recovery, and temporal tracking outputs.")
    add_image_row(document, [pose], ["Sapiens2 308-keypoint pose smoke test"], width=3.2)
    add_caption(document, "Figure 4. Pose estimation preserves the infant box and overlays anatomical keypoints. This is a smoke-test frame, not the final full-dataset result.")

    document.add_heading("Core Mathematical Method", level=1)
    document.add_paragraph("The ROI confidence mechanism combines normalized quality factors:")
    formula = document.add_paragraph()
    formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = formula.add_run("Q(r,t) = w1 Pose + w2 Visibility + w3 Skin + w4 Temporal consistency - w5 Motion")
    r.bold = True
    document.add_paragraph("ROI selection uses a softmax distribution so that reliable regions receive larger weights while weaker regions remain measurable:")
    formula = document.add_paragraph()
    formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = formula.add_run("alpha(r,t) = exp(Q(r,t)) / sum_j exp(Q(j,t))")
    r.bold = True
    add_bullet(document, "The default pilot regime uses equal fixed weights.")
    add_bullet(document, "Validation-tuned and weakly supervised regimes are available, but they use transparent quality proxies rather than fabricated clinical labels.")

    document.add_heading("Ablation Design", level=1)
    ablation_rows = [
        ("B0", "Manual/fixed ROI", "Baseline spatial assumption"),
        ("B1", "Pose + fixed ROI", "Adds pose information"),
        ("B2", "Pose + anatomy + fixed ROI", "Adds anatomical meaning"),
        ("B3", "Pose + anatomy + visibility", "Rejects visually unavailable regions"),
        ("B4", "Pose + anatomy + visibility + motion", "Accounts for nuisance motion"),
        ("B5", "Proposed Module 1", "Dynamic quality-weighted ROI representation"),
    ]
    add_table(document, ["Config.", "Components", "Purpose"], ablation_rows, [0.7, 2.5, 3.9])
    document.add_paragraph("The evaluation reports localization IoU, SNR proxy, confidence-quality correlation, temporal stability, selection consistency, detection metrics, recovery rate, track continuity, and fragmentation. Mean, median, standard deviation, confidence intervals, and subject-level paired tests are generated when statistically valid.")

    document.add_heading("Current Status and Interpretation", level=1)
    status_rows = [
        ("Phases 0-6", "Implemented; detector, fusion, recovery, and tracking outputs generated."),
        ("Phases 7-10", "Implemented; current workspace contains a Sapiens2 smoke test. Full CPU execution is not recommended."),
        ("Phase 11", "Ablation framework implemented; reports generated with explicit availability status."),
        ("Ground truth", "Not currently supplied; truth-dependent metrics are correctly reported as NOT_AVAILABLE."),
        ("Software verification", "230 automated tests passed."),
    ]
    add_table(document, ["Area", "Status"], status_rows, [1.5, 5.6])
    document.add_paragraph(
        "The current report demonstrates a functioning and auditable software pipeline. It should not be presented as final clinical or accuracy validation until the full pose-to-confidence pipeline has been rerun and frame-level ground-truth annotations are available."
    )

    document.add_heading("Next Experimental Steps", level=1)
    add_bullet(document, "Run Sapiens2 on the full dataset with GPU access.")
    add_bullet(document, "Run Phases 8-10 using the complete pose output.")
    add_bullet(document, "Create frame-level ground truth containing infant presence and bounding boxes; add signal-quality annotations where available.")
    add_bullet(document, "Run Phase 11 and compare B0-B5 using the generated CSV reports and figures.")
    add_bullet(document, "Only after Module 1 validation, proceed to physiological signal extraction and HR/RR estimation.")

    document.add_heading("Commands for the Complete Run", level=1)
    commands = (
        "python scripts/run_sapiens2_pose.py --manifest dataset_manifest.csv\n"
        "python scripts/run_anatomical_rois.py --manifest dataset_manifest.csv\n"
        "python scripts/run_roi_quality.py --manifest dataset_manifest.csv\n"
        "python scripts/run_roi_confidence.py --manifest dataset_manifest.csv\n"
        "python scripts/run_ablation_evaluation.py --manifest dataset_manifest.csv --ground-truth path\\to\\ground_truth.csv"
    )
    paragraph = document.add_paragraph()
    run = paragraph.add_run(commands)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)

    document.add_heading("Generated Project Reports", level=1)
    document.add_paragraph("The implementation writes the following Phase 11 files under reports/:")
    add_bullet(document, "ablation_results.csv - per-video metric results")
    add_bullet(document, "ablation_statistics.csv - aggregate statistics")
    add_bullet(document, "statistical_tests.csv - paired tests and corrected p-values")
    add_bullet(document, "figures/ - visual summaries")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Module 1 | Infant-specific visual representation | Presentation report").font.size = Pt(8)
    document.save(REPORT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    build_document()
