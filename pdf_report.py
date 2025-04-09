from datetime import datetime
from fpdf import FPDF
import tempfile
import os

def generate_todays_pdf_report(calc_collection, collection):
    """Generate a PDF report with table statistics for today's date."""
    try:
        # Get today's date in the required format
        date_obj = datetime.today().date().isoformat()
        
        # Collect table stats
        records = list(calc_collection.find({"date": date_obj}))
        time_table = {}
        for record in records:
            name = record.get("name", "Unknown")
            total_frames = record.get("total_frames", 0)
            detected_frames = record.get("detected_frames", 0)
            tot_time = (3 * total_frames) / 60
            det_time = (3 * detected_frames) / 60
            off_time = tot_time - det_time

            time_table[name] = {
                "total_time": round(tot_time, 2),
                "detected_time": round(det_time, 2),
                "off_time": round(off_time, 2),
            }

        # Get uniform data from another collection
        uniform_stats = {}
        uniform_records = list(collection.find({"date": date_obj}))
        for doc in uniform_records:
            detected = doc.get("detected_appearances", 0)
            total = doc.get("total_appearances", 0)
            ratio = round(detected / total, 2) if total else 0
            uniform_stats[doc['name']] = "Yes" if ratio >= 0.4 else "No"

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt=f"Table Statistics for {date_obj}", ln=1, align="C")

        # Table Header
        pdf.cell(50, 10, txt="Name", border=1)
        pdf.cell(30, 10, txt="Total Time", border=1)
        pdf.cell(30, 10, txt="Work Time", border=1)
        pdf.cell(30, 10, txt="Off Time", border=1)
        pdf.cell(30, 10, txt="Uniform", border=1)
        pdf.ln()

        # Table Rows
        for name, stats in time_table.items():
            uniform_status = uniform_stats.get(name, "_")
            pdf.cell(50, 10, txt=name, border=1)
            pdf.cell(30, 10, txt=str(stats["total_time"]), border=1)
            pdf.cell(30, 10, txt=str(stats["detected_time"]), border=1)
            pdf.cell(30, 10, txt=str(stats["off_time"]), border=1)
            pdf.cell(30, 10, txt=uniform_status, border=1)
            pdf.ln()

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"tables_{date_obj}.pdf")
        pdf.output(pdf_path)

        return pdf_path

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None