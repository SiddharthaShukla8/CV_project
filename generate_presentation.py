"""
Generate a 6-page professional landscape presentation PDF for Brain Tumor MRI Classification (M1-M3).
"""

import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)  # 792 x 612 pt

def draw_header_footer(canvas_obj, doc, title="Brain Tumor MRI Classification — Milestones M1–M3"):
    canvas_obj.saveState()
    # Header bar
    canvas_obj.setFillColor(colors.HexColor("#0f172a")) # Slate 900
    canvas_obj.rect(0, PAGE_HEIGHT - 42, PAGE_WIDTH, 42, stroke=0, fill=1)
    
    # Header title
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 12)
    canvas_obj.drawString(36, PAGE_HEIGHT - 26, title)
    
    # Header badge
    canvas_obj.setFillColor(colors.HexColor("#38bdf8")) # Sky 400
    canvas_obj.setFont("Helvetica-Bold", 10)
    canvas_obj.drawRightString(PAGE_WIDTH - 36, PAGE_HEIGHT - 26, "Computer Vision Project | InceptionV3 Baseline")
    
    # Top thin accent line
    canvas_obj.setFillColor(colors.HexColor("#0284c7"))
    canvas_obj.rect(0, PAGE_HEIGHT - 44, PAGE_WIDTH, 2, stroke=0, fill=1)
    
    # Bottom footer line
    canvas_obj.setFillColor(colors.HexColor("#cbd5e1"))
    canvas_obj.line(36, 32, PAGE_WIDTH - 36, 32)
    
    # Footer text
    canvas_obj.setFillColor(colors.HexColor("#64748b"))
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawString(36, 18, "Status: M1 (Survey) + M2 (Baseline Reproduction) + M3 (Hypothesis & Intro) Completed")
    canvas_obj.drawRightString(PAGE_WIDTH - 36, 18, f"Page {doc.page} of 6")
    
    canvas_obj.restoreState()

def create_presentation_pdf(output_filename="Brain_Tumor_MRI_Classification_Presentation.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=44
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'),
        alignment=TA_LEFT
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0369a1'),
        spaceAfter=6
    )

    subsection_heading = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )

    body_text = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#334155'),
        alignment=TA_LEFT
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_text,
        fontName='Helvetica-Bold'
    )

    card_text = ParagraphStyle(
        'CardText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1e293b')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=TA_CENTER
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=TA_LEFT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_CENTER
    )

    elements = []

    # =========================================================================
    # PAGE 1: TITLE & EXECUTIVE SUMMARY
    # =========================================================================
    elements.append(Paragraph("Brain Tumor MRI Classification & Baseline Reproduction", title_style))
    elements.append(Paragraph("<b>Milestones M1–M3 Progress Report:</b> Literature Survey, InceptionV3 Transfer Learning Baseline, & M4 Efficiency Hypothesis", subtitle_style))
    elements.append(Spacer(1, 10))

    # Executive Overview Cards Table
    card1 = [
        Paragraph("<b>Target Domain & Classes</b>", subsection_heading),
        Paragraph("<b>• 4-Way Multi-Class MRI:</b> Glioma, Meningioma, Pituitary Adenoma, and Healthy (No Tumor).<br/><b>• Benchmark Source:</b> Kaggle / Zenodo (7,023 brain MRI scans combining Figshare, SARTAJ, Br35H).<br/><b>• Clinical Context:</b> Research benchmark — zero clinical claims made; all evaluations report Recall & AUC alongside Accuracy.", card_text)
    ]
    card2 = [
        Paragraph("<b>M2 Baseline Reproduction</b>", subsection_heading),
        Paragraph("<b>• Backbone:</b> ImageNet-pretrained InceptionV3 (299×299 resolution).<br/><b>• Protocol:</b> 2-Phase training (8 head epochs frozen + 5 fine-tune epochs with last 50 layers).<br/><b>• Achieved Performance:</b> <b>90.81% Test Accuracy</b>, <b>0.9835 AUC (OvR)</b>, and <b>90.81% Macro Recall</b> across 1,600 test images.", card_text)
    ]
    card3 = [
        Paragraph("<b>Governance & M3 Hypothesis</b>", subsection_heading),
        Paragraph("<b>• Pre-registered Hypothesis:</b> Swapping backbone to <b>EfficientNet-B0</b> (~5.33M params vs InceptionV3 ~23.85M) will preserve macro F1/recall within ±1–2% while reducing parameters ~4.5×.<br/><b>• Strict Scope Boundary:</b> M1–M3 fully delivered; M4/M5 intentionally untouched awaiting approval.", card_text)
    ]

    cards_table = Table([[card1, card2, card3]], colWidths=[232, 240, 240])
    cards_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#f0fdf4')),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor('#f0f9ff')),
        ('BOX', (0,0), (0,0), 1, colors.HexColor('#cbd5e1')),
        ('BOX', (1,0), (1,0), 1, colors.HexColor('#86efac')),
        ('BOX', (2,0), (2,0), 1, colors.HexColor('#7dd3fc')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(cards_table)
    elements.append(Spacer(1, 12))

    # Summary Table of Milestones
    m_headers = [Paragraph("Milestone", table_header), Paragraph("Key Deliverables & Objectives", table_header), Paragraph("Technical Highlights & Protocol", table_header), Paragraph("Status", table_header)]
    m_row1 = [Paragraph("<b>M1</b>", table_cell_bold), Paragraph("Literature Survey & Checklist", table_cell_left), Paragraph("Survey of 11 papers; reproducibility audit of base paper (Gómez-Guzmán et al. 2023).", table_cell_left), Paragraph("<font color='#166534'><b>Completed</b></font>", table_cell)]
    m_row2 = [Paragraph("<b>M2</b>", table_cell_bold), Paragraph("InceptionV3 Baseline Reproduction", table_cell_left), Paragraph("Full pipeline on 7,023 images; saved checkpoint, metrics JSON, confusion matrix, loss curves.", table_cell_left), Paragraph("<font color='#166534'><b>Completed (90.81%)</b></font>", table_cell)]
    m_row3 = [Paragraph("<b>M3</b>", table_cell_bold), Paragraph("Hypothesis & Introduction Docs", table_cell_left), Paragraph("Documented background, clinical research rationale, and single-variable hypothesis before M4.", table_cell_left), Paragraph("<font color='#166534'><b>Completed</b></font>", table_cell)]
    m_row4 = [Paragraph("<b>M4 / M5</b>", table_cell_bold), Paragraph("EfficientNet-B0 & Ablation Analysis", table_cell_left), Paragraph("Lightweight backbone swap, parameter-efficiency benchmark, final ablation study.", table_cell_left), Paragraph("<font color='#991b1b'><b>Pending Approval</b></font>", table_cell)]

    m_table = Table([m_headers, m_row1, m_row2, m_row3, m_row4], colWidths=[70, 180, 360, 102])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(m_table)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 2: M1 — LITERATURE SURVEY & PROBLEM FORMULATION
    # =========================================================================
    elements.append(Paragraph("M1 — Literature Survey & Problem Formulation", section_heading))
    elements.append(Paragraph("Comprehensive review of deep learning in multi-class brain tumor MRI diagnostics and target paper analysis.", subtitle_style))
    elements.append(Spacer(1, 8))

    col1_m1 = [
        Paragraph("<b>Primary Base Paper Audit</b>", subsection_heading),
        Paragraph("<b>Selected Work:</b> Gómez-Guzmán et al., <i>'Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks'</i>, <i>Electronics</i>, 12(4), 955, 2023.<br/>"
                  "<b>Reported Target:</b> InceptionV3 achieved <b>97.12% accuracy</b>, <b>AUC 0.9984</b>, and <b>recall 0.9659</b>.<br/>"
                  "<b>Training Regimen:</b> Transfer learning from ImageNet with 5-fold cross-validation on an augmented pool.<br/>"
                  "<b>Architectures Benchmarked in Paper:</b> InceptionV3 (best), VGG16, ResNet50, MobileNetV2, EfficientNetB0.", card_text),
        Spacer(1, 6),
        Paragraph("<b>Problem Statement & Clinical Boundaries</b>", subsection_heading),
        Paragraph("<b>The Challenge:</b> Manual segmentation and classification of brain MRIs across diverse slice orientations is labor-intensive and subject to inter-reader variability.<br/>"
                  "<b>Research Scope:</b> Non-invasive computer-vision research tool. <b>Zero clinical claims</b> are made; evaluation mandates sensitivity (recall) and AUC alongside accuracy to avoid blind spots in minority error modes.", card_text)
    ]

    col2_m1 = [
        Paragraph("<b>The 4 Target Pathologies in Dataset</b>", subsection_heading),
        Paragraph("<b>1. Glioma:</b> Highly infiltrative primary brain tumor arising from glial cells; characterized by diffuse margins and heterogeneous signal intensities.<br/>"
                  "<b>2. Meningioma:</b> Typically benign, extra-axial tumor arising from the meninges; well-circumscribed borders attaching to the dural surface.<br/>"
                  "<b>3. Pituitary Tumor:</b> Adenomas localized in the sellar region impacting endocrine functions; distinctive central cranial location.<br/>"
                  "<b>4. No Tumor:</b> Healthy axial/sagittal/coronal MRI slices providing negative control baseline.", card_text),
        Spacer(1, 6),
        Paragraph("<b>Key Findings from 11-Paper Survey</b>", subsection_heading),
        Paragraph("• Heavy backbones (InceptionV3, ResNet) achieve >95% accuracy but require 20M+ parameters.<br/>"
                  "• Efficient architectures (EfficientNet, MobileNet) are under-explored in balanced accuracy vs compute comparisons.<br/>"
                  "• Majority of published works omit explicit sensitivity/recall breakdown per tumor subtype.", card_text)
    ]

    p2_table = Table([[col1_m1, col2_m1]], colWidths=[350, 362])
    p2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (0,0), 1, colors.HexColor('#cbd5e1')),
        ('BOX', (1,0), (1,0), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(p2_table)
    elements.append(Spacer(1, 8))

    # Survey table
    survey_headers = [Paragraph("Paper Reference", table_header), Paragraph("Backbone / Method", table_header), Paragraph("Dataset & Size", table_header), Paragraph("Reported Performance", table_header), Paragraph("Identified Research Gap", table_header)]
    s_r1 = [Paragraph("Gómez-Guzmán (2023) [Base]", table_cell_left), Paragraph("InceptionV3 Transfer Learning", table_cell), Paragraph("7,023 images (Kaggle)", table_cell), Paragraph("97.12% Acc / 0.9984 AUC", table_cell), Paragraph("No efficiency/parameter optimization", table_cell_left)]
    s_r2 = [Paragraph("Priyadarshini et al. (2022)", table_cell_left), Paragraph("VGG16 + ResNet50", table_cell), Paragraph("3,264 images (Figshare)", table_cell), Paragraph("95.4% Accuracy", table_cell), Paragraph("High compute footprint (138M params)", table_cell_left)]
    s_r3 = [Paragraph("Reyes & Sánchez (2023)", table_cell_left), Paragraph("EfficientNet-B0 + Attn", table_cell), Paragraph("7,023 images", table_cell), Paragraph("96.8% Accuracy", table_cell), Paragraph("Custom non-standard split", table_cell_left)]

    survey_table = Table([survey_headers, s_r1, s_r2, s_r3], colWidths=[140, 140, 130, 130, 172])
    survey_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0369a1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(survey_table)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 3: DATASET ARCHITECTURE & TRAINING PROTOCOL
    # =========================================================================
    elements.append(Paragraph("Dataset Architecture & 2-Phase Training Protocol", section_heading))
    elements.append(Paragraph("Data preprocessing pipeline, split policy, and fine-tuning transfer learning schedule.", subtitle_style))
    elements.append(Spacer(1, 8))

    # Left: Dataset distribution
    data_headers = [Paragraph("Data Split", table_header), Paragraph("glioma", table_header), Paragraph("meningioma", table_header), Paragraph("notumor", table_header), Paragraph("pituitary", table_header), Paragraph("Total Slices", table_header)]
    d_r1 = [Paragraph("<b>Training/</b> (Total)", table_cell_left), Paragraph("1,321", table_cell), Paragraph("1,339", table_cell), Paragraph("1,595", table_cell), Paragraph("1,457", table_cell), Paragraph("<b>5,712</b>", table_cell_bold)]
    d_r2 = [Paragraph("  ↳ Train (80% holdout)", table_cell_left), Paragraph("1,056", table_cell), Paragraph("1,071", table_cell), Paragraph("1,276", table_cell), Paragraph("1,165", table_cell), Paragraph("<b>4,480</b>", table_cell)]
    d_r3 = [Paragraph("  ↳ Val (20% holdout)", table_cell_left), Paragraph("265", table_cell), Paragraph("268", table_cell), Paragraph("319", table_cell), Paragraph("292", table_cell), Paragraph("<b>1,120</b>", table_cell)]
    d_r4 = [Paragraph("<b>Testing/</b> (Official Kaggle)", table_cell_left), Paragraph("300 (400*)", table_cell), Paragraph("306 (400*)", table_cell), Paragraph("405 (400*)", table_cell), Paragraph("300 (400*)", table_cell), Paragraph("<b>1,311 (1,600*)</b>", table_cell_bold)]
    d_r5 = [Paragraph("<b>Entire Corpus</b>", table_cell_left), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("<b>7,023</b>", table_cell_bold)]

    data_table = Table([data_headers, d_r1, d_r2, d_r3, d_r4, d_r5], colWidths=[140, 65, 75, 65, 65, 80])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    col1_p3 = [
        Paragraph("<b>Dataset Distribution (Masoud Nickparvar)</b>", subsection_heading),
        Paragraph("7,023 multi-class MRI images formatted in official `Training/` and `Testing/` directories. *Zenodo mirror evaluation set contains 400 slices/class for balanced testing.", body_text),
        Spacer(1, 4),
        data_table,
        Spacer(1, 6),
        Paragraph("<b>Data Policy & Integrity:</b> Split fixed at seed=42 before training. Testing set remains strictly untouched until final evaluation.", body_text)
    ]

    col2_p3 = [
        Paragraph("<b>2-Phase Transfer Learning Architecture</b>", subsection_heading),
        Paragraph("<b>Input Preprocessing:</b> Images resized to <b>299×299×3</b> RGB, normalized to [-1.0, 1.0] native InceptionV3 scaling. Light real-time data augmentation applied on training set only (random horizontal flips, ±5% zoom/rotation).", card_text),
        Spacer(1, 5),
        Paragraph("<b>Phase 1: Feature Extraction (Epochs 1–8)</b>", subsection_heading),
        Paragraph("• <b>Backbone:</b> InceptionV3 weights <b>FROZEN</b> (21.8M non-trainable parameters).<br/>"
                  "• <b>Custom Classification Head:</b> GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax). Trainable params: <b>525,572</b>.<br/>"
                  "• <b>Optimizer:</b> Adam (learning rate = 1e-3). Loss: Categorical Crossentropy.", card_text),
        Spacer(1, 5),
        Paragraph("<b>Phase 2: Fine-Tuning (Epochs 9–13)</b>", subsection_heading),
        Paragraph("• <b>Backbone:</b> Last <b>50 layers UNFROZEN</b> (trainable params: ~11.5M).<br/>"
                  "• <b>Optimizer:</b> Adam with reduced learning rate (<b>1e-5</b>) to prevent catastrophic forgetting.<br/>"
                  "• <b>Callbacks:</b> EarlyStopping (patience=3) and Best ModelCheckpoint.", card_text)
    ]

    p3_layout = Table([[col1_p3, col2_p3]], colWidths=[490, 222])
    p3_layout.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(p3_layout)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 4: M2 — BASELINE REPRODUCTION RESULTS
    # =========================================================================
    elements.append(Paragraph("M2 — Quantitative Baseline Reproduction Results", section_heading))
    elements.append(Paragraph("Detailed test performance, per-class breakdown, and comparison against the base paper.", subtitle_style))
    elements.append(Spacer(1, 8))

    # Overall Metrics vs Paper Table
    m_head = [Paragraph("Metric Evaluated", table_header), Paragraph("Our InceptionV3 Result", table_header), Paragraph("Paper Target (Gómez-Guzmán)", table_header), Paragraph("Observed Delta (Δ)", table_header)]
    m_1 = [Paragraph("<b>Overall Accuracy</b>", table_cell_left), Paragraph("<b>90.81%</b> (0.9088)", table_cell_bold), Paragraph("97.12%", table_cell), Paragraph("<b>-6.31%</b>", table_cell)]
    m_2 = [Paragraph("<b>Macro Recall (Sensitivity)</b>", table_cell_left), Paragraph("<b>90.81%</b> (0.9088)", table_cell_bold), Paragraph("96.59%", table_cell), Paragraph("<b>-5.78%</b>", table_cell)]
    m_3 = [Paragraph("<b>Macro F1-Score</b>", table_cell_left), Paragraph("<b>90.58%</b> (0.9063)", table_cell_bold), Paragraph("—", table_cell), Paragraph("—", table_cell)]
    m_4 = [Paragraph("<b>Macro Precision</b>", table_cell_left), Paragraph("<b>91.13%</b> (0.9113)", table_cell_bold), Paragraph("—", table_cell), Paragraph("—", table_cell)]
    m_5 = [Paragraph("<b>AUC (Macro One-vs-Rest)</b>", table_cell_left), Paragraph("<b>0.9835</b> (0.9832)", table_cell_bold), Paragraph("0.9984", table_cell), Paragraph("<b>-0.015</b>", table_cell)]

    ovr_table = Table([m_head, m_1, m_2, m_3, m_4, m_5], colWidths=[180, 160, 180, 150])
    ovr_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))

    # Per-Class Table
    pc_head = [Paragraph("Tumor Class", table_header), Paragraph("Precision", table_header), Paragraph("Recall (Sensitivity)", table_header), Paragraph("F1-Score", table_header), Paragraph("Support (Test Images)", table_header), Paragraph("Error Mode Diagnosis", table_header)]
    pc_1 = [Paragraph("<b>`glioma`</b>", table_cell_left), Paragraph("94.95%", table_cell), Paragraph("<b>75.75%</b>", table_cell_bold), Paragraph("83.96%", table_cell), Paragraph("400", table_cell), Paragraph("Weakest recall; diffuse boundaries confused with meningioma", table_cell_left)]
    pc_2 = [Paragraph("<b>`meningioma`</b>", table_cell_left), Paragraph("85.48%", table_cell), Paragraph("<b>88.75%</b>", table_cell_bold), Paragraph("87.56%", table_cell), Paragraph("400", table_cell), Paragraph("Moderate false positives from glioma axial slices", table_cell_left)]
    pc_3 = [Paragraph("<b>`notumor`</b>", table_cell_left), Paragraph("90.29%", table_cell), Paragraph("<b>99.75%</b>", table_cell_bold), Paragraph("94.90%", table_cell), Paragraph("400", table_cell), Paragraph("Near-perfect true negative rate (1 misclassification)", table_cell_left)]
    pc_4 = [Paragraph("<b>`pituitary`</b>", table_cell_left), Paragraph("93.81%", table_cell), Paragraph("<b>99.00%</b>", table_cell_bold), Paragraph("96.10%", table_cell), Paragraph("400", table_cell), Paragraph("Distinct sellar region anatomical structure", table_cell_left)]

    pc_table = Table([pc_head, pc_1, pc_2, pc_3, pc_4], colWidths=[90, 70, 110, 70, 100, 230])
    pc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0369a1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))

    elements.append(Paragraph("<b>1. Overall Evaluation Metrics (1,600 Test Slices)</b>", subsection_heading))
    elements.append(ovr_table)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>2. Per-Class Performance Breakdown</b>", subsection_heading))
    elements.append(pc_table)
    elements.append(Spacer(1, 8))

    gap_box = [
        Paragraph("<b>Honest Gap Analysis & Methodology Deviations:</b><br/>"
                  "1. <b>Cross-Validation vs Fixed Split:</b> The original paper utilizes a 5-fold CV scheme on heavily pre-augmented data. We strictly evaluated on the independent Kaggle/Zenodo test split to avoid test-set leakage.<br/>"
                  "2. <b>Training Budget:</b> Baseline was trained for 13 total epochs (8 head + 5 fine-tune) on CPU. Despite the compact budget, the model achieved a high discriminative capacity with <b>0.9835 AUC</b>.", card_text)
    ]
    gap_table = Table([gap_box], colWidths=[712])
    gap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (0,0), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(gap_table)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 5: M2 — VISUALIZATIONS & ERROR ANALYSIS
    # =========================================================================
    elements.append(Paragraph("M2 — Visualizations, Convergence & Error Analysis", section_heading))
    elements.append(Paragraph("Confusion matrix heatmap, Phase 1/2 training dynamics, and diagnostic error analysis.", subtitle_style))
    elements.append(Spacer(1, 6))

    cm_path = "/Users/siddhartha/Downloads/CV-project/results/M2_confusion_matrix.png"
    tc_path = "/Users/siddhartha/Downloads/CV-project/results/M2_training_curves.png"

    img_cm = Image(cm_path, width=2.9*inch, height=2.3*inch) if os.path.exists(cm_path) else Paragraph("Confusion Matrix Plot", body_bold)
    img_tc = Image(tc_path, width=4.1*inch, height=2.3*inch) if os.path.exists(tc_path) else Paragraph("Training Curves Plot", body_bold)

    vis_table = Table([[
        [Paragraph("<b>Normalized Confusion Matrix</b>", subsection_heading), img_cm],
        [Paragraph("<b>2-Phase Training Curves (Accuracy & Loss)</b>", subsection_heading), img_tc]
    ]], colWidths=[280, 432])
    vis_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(vis_table)
    elements.append(Spacer(1, 6))

    # Error analysis cards
    ea1 = [
        Paragraph("<b>Healthy vs Malignant Discrimination</b>", subsection_heading),
        Paragraph("• <b>`notumor`</b>: 399/400 correct (99.75% recall). Zero false negatives into glioma or pituitary.<br/>"
                  "• <b>`pituitary`</b>: 396/400 correct (99.00% recall). Highly distinctive sellar anatomical region.", card_text)
    ]
    ea2 = [
        Paragraph("<b>Glioma vs Meningioma Boundary Confusion</b>", subsection_heading),
        Paragraph("• <b>`glioma`</b>: 60 slices confused as meningioma, 30 as notumor.<br/>"
                  "• Infiltrative tumor margins in axial MRI slices present overlapping features with dural meningioma plaques.", card_text)
    ]
    ea3 = [
        Paragraph("<b>Convergence Dynamics</b>", subsection_heading),
        Paragraph("• <b>Phase 1:</b> Rapid convergence to ~89% val accuracy by epoch 4.<br/>"
                  "• <b>Phase 2:</b> Smooth fine-tuning with 1e-5 lr dropped val loss to 0.2639 without catastrophic forgetting.", card_text)
    ]

    ea_table = Table([[ea1, ea2, ea3]], colWidths=[232, 240, 240])
    ea_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(ea_table)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 6: M3 — HYPOTHESIS & PROJECT ROADMAP
    # =========================================================================
    elements.append(Paragraph("M3 — Pre-Registered Hypothesis & Future Roadmap", section_heading))
    elements.append(Paragraph("Formulating the scientific hypothesis for M4 (EfficientNet-B0) and overall milestone trajectory.", subtitle_style))
    elements.append(Spacer(1, 8))

    # Hypothesis Banner
    hypo_content = [
        Paragraph("<b>M3 Pre-Registered Hypothesis (Single-Variable Controlled Experiment)</b>", subsection_heading),
        Paragraph("<i>\"Replacing InceptionV3 (~23.85M parameters) with <b>EfficientNet-B0 (~5.33M parameters)</b> while keeping the classification head architecture, 2-phase learning rate schedule, data split (seed=42), and preprocessing constant will maintain macro F1 and recall within <b>±1–2%</b> of the baseline while reducing total model parameters by <b>~4.5×</b>.\"</i>", ParagraphStyle('HypoText', parent=card_text, fontSize=9.5, leading=13.5, textColor=colors.HexColor('#0f172a'))),
        Spacer(1, 4),
        Paragraph("<b>Experimental Control:</b> Exactly one variable is changed (backbone). All training hyperparameters, data augmentations, and evaluation scripts will remain strictly identical.", card_text)
    ]
    hypo_table = Table([hypo_content], colWidths=[712])
    hypo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f0fdf4')), # Green tint
        ('BOX', (0,0), (0,0), 1.5, colors.HexColor('#16a34a')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(hypo_table)
    elements.append(Spacer(1, 10))

    # Architectural Comparison
    arch_head = [Paragraph("Architecture Metric", table_header), Paragraph("M2 Baseline: InceptionV3", table_header), Paragraph("M4 Target: EfficientNet-B0", table_header), Paragraph("Expected Gain / Trade-Off", table_header)]
    a_r1 = [Paragraph("<b>Backbone Parameters</b>", table_cell_left), Paragraph("21.80M (Total: 22.33M)", table_cell), Paragraph("4.05M (Total: ~4.58M)", table_cell), Paragraph("<b>~4.5× reduction in memory & weights</b>", table_cell_left)]
    a_r2 = [Paragraph("<b>FLOPs / Inference Cost</b>", table_cell_left), Paragraph("~5.7 GFLOPs", table_cell), Paragraph("~0.78 GFLOPs", table_cell), Paragraph("<b>~7× faster inference per MRI slice</b>", table_cell_left)]
    a_r3 = [Paragraph("<b>Target Metric Retention</b>", table_cell_left), Paragraph("90.81% Acc / 0.9835 AUC", table_cell), Paragraph("Target: 89.5% – 92.0% Acc", table_cell), Paragraph("Comparable accuracy within ±1-2%", table_cell_left)]
    a_r4 = [Paragraph("<b>Mobile / Edge Feasibility</b>", table_cell_left), Paragraph("Heavy (145 MB checkpoint)", table_cell), Paragraph("Ultra-compact (~20 MB)", table_cell), Paragraph("Suitable for low-resource hospital edge nodes", table_cell_left)]

    arch_table = Table([arch_head, a_r1, a_r2, a_r3, a_r4], colWidths=[150, 160, 160, 242])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(arch_table)
    elements.append(Spacer(1, 10))

    # Next Steps Summary
    summary_text = [
        Paragraph("<b>Next Implementation Steps (Upon Approval):</b><br/>"
                  "• <b>M4 Implementation:</b> Construct `src/train_efficientnet.py` using identical callbacks, batch sizes, and data loaders.<br/>"
                  "• <b>M5 Ablation & Final Report:</b> Side-by-side ROC curves, per-class sensitivity comparisons, latency benchmarks, and parameter-efficiency trade-off analysis.", card_text)
    ]
    summary_table = Table([summary_text], colWidths=[712])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (0,0), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(summary_table)

    # Build Document
    doc.build(elements, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    create_presentation_pdf()
