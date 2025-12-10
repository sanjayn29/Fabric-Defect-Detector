# app.py
from flask import Flask, render_template, Response, redirect, make_response
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
import urllib.parse

# PDF Generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

app = Flask(__name__)

# ====================== DATABASE ======================
DATABASE_URL = "postgresql://postgres:N.Sanjay%402005@localhost:5432/Fab"
url = urlparse(DATABASE_URL)
db_config = {
    'dbname': url.path[1:],
    'user': url.username,
    'password': urllib.parse.unquote(url.password),
    'host': url.hostname,
    'port': url.port or 5432
}

def get_db_connection():
    return psycopg2.connect(**db_config)

# ====================== SETUP ======================
SAVE_DIR = "static/defects"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = 'best_model.h5'
CLASS_NAMES = ['broken_stitch', 'defect-free', 'hole', 'hole', 'lines',
               'needle_mark', 'pinched_fabric', 'stain', 'stain']
INPUT_SIZE = (224, 224)
DEFECT_THRESHOLD = 0.85

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

last_save_time = 0
MIN_SAVE_INTERVAL = 3  # seconds

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(INPUT_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_with_tta(frame):
    preds = []
    for flip in [-1, 0, 1]:  # original + horizontal + vertical flip
        img = frame if flip == -1 else cv2.flip(frame, flip)
        preds.append(model.predict(preprocess_image(img), verbose=0)[0])
    avg = np.mean(preds, axis=0)
    return CLASS_NAMES[np.argmax(avg)], np.max(avg)

def save_defect(defect_name, confidence, image_path):
    global last_save_time
    if time.time() - last_save_time < MIN_SAVE_INTERVAL:
        return False
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO defects (defect_name, confidence, image_path) VALUES (%s, %s, %s)",
            (defect_name, float(confidence), image_path)
        )
        conn.commit()
        cur.close()
        conn.close()
        last_save_time = time.time()
        print(f"[SAVED] {defect_name} ({confidence:.1%})")
        return True
    except Exception as e:
        print("DB Error:", e)
        return False

# ====================== CAMERA STREAM ======================
def generate_frames():
    # Try multiple backends (Windows fix)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback

    if not cap.isOpened():
        print("Camera not accessible! Serving placeholder.")
        placeholder = open("static/no_camera.jpg", "rb").read()
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)

    print("Camera started successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_class, confidence = predict_with_tta(frame)
        color = (0, 255, 0) if pred_class == 'defect-free' else (0, 0, 255)
        label = f"{pred_class.replace('_', ' ').title()} ({confidence:.1%})"
        cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

        if pred_class != 'defect-free' and confidence > DEFECT_THRESHOLD:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
            cv2.putText(frame, "DEFECT DETECTED!", (frame.shape[1]//4, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 0, 255), 7)

            filename = f"{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            full_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(full_path, frame)
            web_path = f"/{SAVE_DIR}/{filename}"
            save_defect(pred_class, confidence, web_path)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ====================== ROUTES ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/defect_count')
def defect_count():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM defects")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"count": count}
    except:
        return {"count": 0}

@app.route('/defects')
def defects_list():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM defects ORDER BY detected_at DESC")
        defects = cur.fetchall()

        cur.execute("SELECT defect_name, COUNT(*) as count FROM defects GROUP BY defect_name ORDER BY count DESC")
        stats = cur.fetchall()
        labels = [row['defect_name'].replace('_', ' ').title() for row in stats]
        values = [row['count'] for row in stats]

        cur.close()
        conn.close()
    except Exception as e:
        print("Error:", e)
        defects, labels, values = [], [], []

    return render_template('defects.html',
                          defects=defects,
                          defect_stats={'labels': labels, 'values': values})

@app.route('/delete_defect/<int:defect_id>')
def delete_defect(defect_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM defects WHERE id = %s", (defect_id,))
        row = cur.fetchone()
        if row:
            path = row[0].lstrip('/')
            if os.path.exists(path):
                os.remove(path)
            cur.execute("DELETE FROM defects WHERE id = %s", (defect_id,))
            conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(e)
    return redirect('/defects')

@app.route('/download_pdf')
def download_pdf():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT defect_name, confidence, image_path, detected_at FROM defects ORDER BY detected_at DESC")
        defects = cur.fetchall()
        cur.close()
        conn.close()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("FABRIC DEFECT REPORT", styles['Title']))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y at %H:%M')}", styles['Normal']))
        elements.append(Paragraph(f"Total Defects Detected: <b>{len(defects)}</b>", styles['Normal']))
        elements.append(Spacer(1, 30))

        if defects:
            data = [["#", "Image", "Defect Type", "Confidence", "Detected At"]]
            for i, d in enumerate(defects, 1):
                img_path = os.path.join(os.getcwd(), d['image_path'].lstrip('/'))
                try:
                    img = RLImage(img_path, width=1.5*inch, height=1.2*inch)
                except:
                    img = Paragraph("[No Image]", styles['Normal'])

                data.append([
                    str(i),
                    img,
                    d['defect_name'].replace('_', ' ').title(),
                    f"{d['confidence']*100:.1f}%",
                    d['detected_at'].strftime('%d %b %Y, %H:%M')
                ])

            table = Table(data, colWidths=[0.5*inch, 1.7*inch, 2*inch, 1*inch, 1.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0066cc')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (1,1), (-1,-1), colors.HexColor('#f8f9fa')),
                ('GRID', (0,0), (-1,-1), 1, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No defects found.", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)

        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Fabric_Defect_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        return response
    except Exception as e:
        print("PDF Error:", e)
        return "Error generating PDF", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)