"""
Report Generation Module for Defect Detection System

Generates PDF reports with prediction results, heatmaps, and severity assessments
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                                Spacer, Image, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from config.config import *


class ReportGenerator:
    """
    Generates professional PDF reports for defect detection results
    """
    
    def __init__(self, output_dir=OUTPUTS_DIR):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Info style
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        ))
    
    def generate_single_image_report(self, result, image_path, heatmap_path=None, 
                                     output_filename=None):
        """
        Generate PDF report for a single image prediction
        
        Args:
            result: Prediction result dictionary
            image_path: Path to original image
            heatmap_path: Path to heatmap image (optional)
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to generated PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"defect_report_{timestamp}.pdf"
        
        pdf_path = os.path.join(self.output_dir, output_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        elements = []
        
        # Title
        elements.append(Paragraph("Defect Detection Report", self.styles['CustomTitle']))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['CustomSubtitle']
        ))
        elements.append(Spacer(1, 20))
        
        # Analysis Result Section
        elements.append(Paragraph("Analysis Result", self.styles['SectionHeader']))
        
        # Determine status
        is_defective = result.get('is_defective', result['predicted_class'] != 'no_defect')
        status = "DEFECTIVE" if is_defective else "GOOD"
        status_color = colors.red if is_defective else colors.green
        
        # Status table
        status_data = [
            ['Status:', status],
            ['Predicted Class:', result['predicted_class'].replace('_', ' ').title()],
            ['Confidence:', f"{result['confidence']:.1%}"],
        ]
        
        # Add severity if present
        if 'severity_score' in result:
            status_data.append(['Severity Score:', f"{result['severity_score']:.1f}/10"])
            status_data.append(['Severity Level:', result.get('severity_level', 'N/A')])
        
        status_table = Table(status_data, colWidths=[2*inch, 4*inch])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (1, 0), (1, 0), status_color),
            ('FONT', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(status_table)
        elements.append(Spacer(1, 20))
        
        # Prediction Details Section
        elements.append(Paragraph("Top Predictions", self.styles['SectionHeader']))
        
        pred_data = [['Rank', 'Class', 'Probability']]
        for i, pred in enumerate(result['predictions'][:5], 1):
            pred_data.append([
                str(i),
                pred['class'].replace('_', ' ').title(),
                f"{pred['probability']:.1%}"
            ])
        
        pred_table = Table(pred_data, colWidths=[0.75*inch, 3*inch, 2.25*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(pred_table)
        elements.append(Spacer(1, 20))
        
        # Images Section
        elements.append(Paragraph("Visual Analysis", self.styles['SectionHeader']))
        
        # Prepare images
        image_elements = []
        
        # Original image
        if os.path.exists(image_path):
            try:
                img = Image(image_path, width=2.5*inch, height=2.5*inch, kind='proportional')
                img.hAlign = 'CENTER'
                image_elements.append(['Original Image', 'Heatmap Overlay' if heatmap_path else ''])
                image_elements.append([img, ''])
            except Exception as e:
                print(f"Warning: Could not add image to report: {e}")
        
        # Heatmap overlay
        if heatmap_path and os.path.exists(heatmap_path):
            try:
                heatmap_img = Image(heatmap_path, width=2.5*inch, height=2.5*inch, kind='proportional')
                heatmap_img.hAlign = 'CENTER'
                if image_elements:
                    image_elements[1][1] = heatmap_img
                else:
                    image_elements.append(['', 'Heatmap Overlay'])
                    image_elements.append(['', heatmap_img])
            except Exception as e:
                print(f"Warning: Could not add heatmap to report: {e}")
        
        if image_elements:
            img_table = Table(image_elements, colWidths=[3*inch, 3*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ]))
            elements.append(img_table)
        
        elements.append(Spacer(1, 20))
        
        # Severity Breakdown (if available)
        if 'severity_breakdown' in result:
            elements.append(Paragraph("Severity Score Breakdown", self.styles['SectionHeader']))
            
            breakdown = result['severity_breakdown']
            breakdown_data = [
                ['Component', 'Score'],
                ['Defect Type Weight', f"{breakdown.get('type_component', 0):.1f}"],
                ['Confidence Factor', f"{breakdown.get('confidence_component', 0):.1f}"],
                ['Heatmap Concentration', f"{breakdown.get('concentration_component', 0):.1f}"],
                ['Total Severity Score', f"{result['severity_score']:.1f}"]
            ]
            
            breakdown_table = Table(breakdown_data, colWidths=[3*inch, 3*inch])
            breakdown_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ffffcc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f0f0f0')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(breakdown_table)
        
        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "Generated by Intelligent Defect Detection System",
            self.styles['InfoText']
        ))
        elements.append(Paragraph(
            f"Model: {DEFAULT_MODEL} | Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}",
            self.styles['InfoText']
        ))
        
        # Build PDF
        doc.build(elements)
        
        print(f"✅ PDF report generated: {pdf_path}")
        return pdf_path
    
    def generate_batch_report(self, results, output_filename=None):
        """
        Generate PDF report for batch predictions
        
        Args:
            results: List of prediction results
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to generated PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"batch_report_{timestamp}.pdf"
        
        pdf_path = os.path.join(self.output_dir, output_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        elements = []
        
        # Title
        elements.append(Paragraph("Batch Defect Detection Report", self.styles['CustomTitle']))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['CustomSubtitle']
        ))
        elements.append(Spacer(1, 20))
        
        # Summary Statistics
        total = len(results)
        defective = sum(1 for r in results if r.get('is_defective', r['predicted_class'] != 'no_defect'))
        good = total - defective
        defect_rate = (defective / total * 100) if total > 0 else 0
        
        elements.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Images Analyzed', str(total)],
            ['Defective Products', str(defective)],
            ['Good Products', str(good)],
            ['Defect Rate', f"{defect_rate:.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Detailed Results
        elements.append(Paragraph("Detailed Results", self.styles['SectionHeader']))
        
        # Table headers
        detail_data = [['#', 'Image', 'Prediction', 'Confidence', 'Severity', 'Status']]
        
        for i, result in enumerate(results[:50], 1):  # Limit to 50 for PDF size
            is_defective = result.get('is_defective', result['predicted_class'] != 'no_defect')
            status = "⚠️ DEFECTIVE" if is_defective else "✅ GOOD"
            
            filename = result.get('filename', result.get('image_path', 'N/A'))
            if isinstance(filename, str):
                filename = os.path.basename(filename)
            
            detail_data.append([
                str(i),
                filename[:20] + '...' if len(filename) > 20 else filename,
                result['predicted_class'].replace('_', ' ').title()[:15],
                f"{result['confidence']:.0%}",
                f"{result.get('severity_score', 0):.0f}" if 'severity_score' in result else 'N/A',
                status
            ])
        
        detail_table = Table(detail_data, colWidths=[0.4*inch, 1.8*inch, 1.5*inch, 0.9*inch, 0.9*inch, 1*inch])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(detail_table)
        
        if len(results) > 50:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(
                f"Note: Showing first 50 of {len(results)} total results",
                self.styles['InfoText']
            ))
        
        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "Generated by Intelligent Defect Detection System",
            self.styles['InfoText']
        ))
        
        # Build PDF
        doc.build(elements)
        
        print(f"✅ Batch PDF report generated: {pdf_path}")
        return pdf_path
