"""
Report Generation Module
Generates PDF and Excel reports for detection events
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import io

# Excel support
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ReportGenerator] pandas not available. Install with: pip install pandas openpyxl")

# PDF support
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("[ReportGenerator] reportlab not available. Install with: pip install reportlab")


class ReportGenerator:
    """
    Generate security reports in PDF and Excel formats

    Reports include:
    - Detection summary (fights, weapons, falls, screams)
    - Timeline of events
    - Statistics and charts
    - Screenshots of incidents
    """

    def __init__(
        self,
        output_dir: str = "reports",
        company_name: str = "ToBeLess AI Security",
        debug: bool = False
    ):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save reports
            company_name: Company name for report header
            debug: Enable debug output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.company_name = company_name
        self.debug = debug

        if self.debug:
            print(f"[ReportGenerator] Initialized. Output dir: {self.output_dir}")
            print(f"[ReportGenerator] pandas available: {PANDAS_AVAILABLE}")
            print(f"[ReportGenerator] reportlab available: {REPORTLAB_AVAILABLE}")

    def generate_excel_report(
        self,
        events: List[Dict],
        statistics: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate Excel report

        Args:
            events: List of detection events
            statistics: Detection statistics
            start_time: Report start time
            end_time: Report end time
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated file or None on failure
        """
        if not PANDAS_AVAILABLE:
            if self.debug:
                print("[ReportGenerator] Cannot generate Excel: pandas not available")
            return None

        try:
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"security_report_{timestamp}.xlsx"

            filepath = self.output_dir / filename

            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_data = {
                    'Metric': [
                        'Report Generated',
                        'Report Period Start',
                        'Report Period End',
                        'Total Events',
                        'Fight Detections',
                        'Weapon Detections',
                        'Fall Detections',
                        'Scream Detections'
                    ],
                    'Value': [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "N/A",
                        end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else "N/A",
                        len(events),
                        statistics.get('fight_detections', 0),
                        statistics.get('weapon_detections', 0),
                        statistics.get('fall_detections', 0),
                        statistics.get('scream_detections', 0)
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Sheet 2: Events Timeline
                if events:
                    events_data = []
                    for event in events:
                        events_data.append({
                            'Timestamp': event.get('timestamp', ''),
                            'Event Type': event.get('type', 'unknown'),
                            'Confidence': f"{event.get('confidence', 0) * 100:.1f}%",
                            'Details': event.get('details', ''),
                            'Location': event.get('location', 'Camera 1'),
                            'Status': event.get('status', 'recorded')
                        })
                    df_events = pd.DataFrame(events_data)
                    df_events.to_excel(writer, sheet_name='Events', index=False)

                # Sheet 3: Hourly Statistics
                hourly_stats = self._calculate_hourly_stats(events)
                if hourly_stats:
                    df_hourly = pd.DataFrame(hourly_stats)
                    df_hourly.to_excel(writer, sheet_name='Hourly Stats', index=False)

                # Sheet 4: Event Types Distribution
                type_stats = self._calculate_type_stats(events)
                if type_stats:
                    df_types = pd.DataFrame(type_stats)
                    df_types.to_excel(writer, sheet_name='Event Types', index=False)

            if self.debug:
                print(f"[ReportGenerator] ✓ Excel report saved: {filepath}")

            return str(filepath)

        except Exception as e:
            if self.debug:
                print(f"[ReportGenerator] Excel generation error: {e}")
            return None

    def generate_pdf_report(
        self,
        events: List[Dict],
        statistics: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filename: Optional[str] = None,
        include_screenshots: bool = True
    ) -> Optional[str]:
        """
        Generate PDF report

        Args:
            events: List of detection events
            statistics: Detection statistics
            start_time: Report start time
            end_time: Report end time
            filename: Output filename (auto-generated if None)
            include_screenshots: Include event screenshots

        Returns:
            Path to generated file or None on failure
        """
        if not REPORTLAB_AVAILABLE:
            if self.debug:
                print("[ReportGenerator] Cannot generate PDF: reportlab not available")
            return None

        try:
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"security_report_{timestamp}.pdf"

            filepath = self.output_dir / filename

            # Create PDF document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#4B0082')
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.HexColor('#333333')
            )
            normal_style = styles['Normal']

            # Build document content
            content = []

            # Title
            content.append(Paragraph(self.company_name, title_style))
            content.append(Paragraph("Security Incident Report", heading_style))
            content.append(Spacer(1, 20))

            # Report info
            report_info = f"""
            <b>Report Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
            <b>Period:</b> {start_time.strftime("%Y-%m-%d %H:%M") if start_time else "N/A"} -
            {end_time.strftime("%Y-%m-%d %H:%M") if end_time else "N/A"}<br/>
            <b>Total Events:</b> {len(events)}
            """
            content.append(Paragraph(report_info, normal_style))
            content.append(Spacer(1, 30))

            # Summary Statistics Table
            content.append(Paragraph("Detection Summary", heading_style))

            summary_data = [
                ['Detection Type', 'Count', 'Status'],
                ['Fight Detection', str(statistics.get('fight_detections', 0)), '●' if statistics.get('fight_detections', 0) > 0 else '○'],
                ['Weapon Detection', str(statistics.get('weapon_detections', 0)), '●' if statistics.get('weapon_detections', 0) > 0 else '○'],
                ['Fall Detection', str(statistics.get('fall_detections', 0)), '●' if statistics.get('fall_detections', 0) > 0 else '○'],
                ['Scream Detection', str(statistics.get('scream_detections', 0)), '●' if statistics.get('scream_detections', 0) > 0 else '○'],
            ]

            summary_table = Table(summary_data, colWidths=[200, 100, 80])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4B0082')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            content.append(summary_table)
            content.append(Spacer(1, 30))

            # Events Timeline
            if events:
                content.append(Paragraph("Events Timeline", heading_style))

                # Limit to recent 50 events for PDF
                recent_events = events[-50:] if len(events) > 50 else events

                events_data = [['Time', 'Type', 'Confidence', 'Details']]
                for event in recent_events:
                    timestamp = event.get('timestamp', '')
                    if isinstance(timestamp, str) and 'T' in timestamp:
                        timestamp = timestamp.split('T')[1][:8]

                    events_data.append([
                        timestamp,
                        event.get('type', 'unknown').upper(),
                        f"{event.get('confidence', 0) * 100:.0f}%",
                        event.get('details', '')[:30]
                    ])

                events_table = Table(events_data, colWidths=[80, 100, 80, 180])
                events_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
                ]))
                content.append(events_table)
                content.append(Spacer(1, 20))

                if len(events) > 50:
                    content.append(Paragraph(
                        f"<i>Showing last 50 of {len(events)} events. Full data available in Excel report.</i>",
                        normal_style
                    ))

            # Footer
            content.append(Spacer(1, 40))
            footer_text = f"""
            <br/><br/>
            <i>This report was automatically generated by {self.company_name}.<br/>
            For questions or concerns, please contact your security administrator.</i>
            """
            content.append(Paragraph(footer_text, normal_style))

            # Build PDF
            doc.build(content)

            if self.debug:
                print(f"[ReportGenerator] ✓ PDF report saved: {filepath}")

            return str(filepath)

        except Exception as e:
            if self.debug:
                print(f"[ReportGenerator] PDF generation error: {e}")
            return None

    def generate_json_report(
        self,
        events: List[Dict],
        statistics: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate JSON report (always available)

        Args:
            events: List of detection events
            statistics: Detection statistics
            start_time: Report start time
            end_time: Report end time
            filename: Output filename

        Returns:
            Path to generated file or None on failure
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"security_report_{timestamp}.json"

            filepath = self.output_dir / filename

            report_data = {
                'meta': {
                    'generated_at': datetime.now().isoformat(),
                    'company': self.company_name,
                    'period_start': start_time.isoformat() if start_time else None,
                    'period_end': end_time.isoformat() if end_time else None
                },
                'statistics': statistics,
                'events': events,
                'hourly_stats': self._calculate_hourly_stats(events),
                'type_stats': self._calculate_type_stats(events)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            if self.debug:
                print(f"[ReportGenerator] ✓ JSON report saved: {filepath}")

            return str(filepath)

        except Exception as e:
            if self.debug:
                print(f"[ReportGenerator] JSON generation error: {e}")
            return None

    def _calculate_hourly_stats(self, events: List[Dict]) -> List[Dict]:
        """Calculate hourly statistics from events"""
        hourly = {}

        for event in events:
            timestamp = event.get('timestamp', '')
            if isinstance(timestamp, str) and 'T' in timestamp:
                hour = timestamp.split('T')[1][:2] + ":00"
            else:
                continue

            if hour not in hourly:
                hourly[hour] = {'hour': hour, 'total': 0, 'fights': 0, 'weapons': 0, 'falls': 0, 'screams': 0}

            hourly[hour]['total'] += 1
            event_type = event.get('type', '').lower()
            if 'fight' in event_type:
                hourly[hour]['fights'] += 1
            elif 'weapon' in event_type:
                hourly[hour]['weapons'] += 1
            elif 'fall' in event_type:
                hourly[hour]['falls'] += 1
            elif 'scream' in event_type:
                hourly[hour]['screams'] += 1

        return list(hourly.values())

    def _calculate_type_stats(self, events: List[Dict]) -> List[Dict]:
        """Calculate event type statistics"""
        type_counts = {}

        for event in events:
            event_type = event.get('type', 'unknown')
            if event_type not in type_counts:
                type_counts[event_type] = 0
            type_counts[event_type] += 1

        return [{'type': k, 'count': v} for k, v in type_counts.items()]

    def get_available_reports(self) -> List[Dict]:
        """Get list of available reports in output directory"""
        reports = []

        for filepath in self.output_dir.glob("security_report_*"):
            stat = filepath.stat()
            reports.append({
                'filename': filepath.name,
                'path': str(filepath),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'type': filepath.suffix[1:].upper()
            })

        return sorted(reports, key=lambda x: x['created'], reverse=True)
