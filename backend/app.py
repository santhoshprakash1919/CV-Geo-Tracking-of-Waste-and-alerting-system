from flask import Flask, request, jsonify, send_from_directory
import sqlite3, os, datetime
from ai_model.predict import predict
from backend.database import init_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DB_PATH = os.path.join(BASE_DIR, 'database', 'garbage.db')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    lat = request.form.get('lat')
    lng = request.form.get('lng')

    if not file.filename:
        return jsonify({'error': 'Image filename is empty'}), 400

    filename = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    result = predict(save_path)
    if 'error' in result:
        return jsonify(result), 503

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO detections (image_path, prediction, confidence, latitude, longitude, timestamp) VALUES (?,?,?,?,?,?)',
        (save_path, result['prediction'], result['confidence'], lat, lng, str(datetime.datetime.now()))
    )
    conn.commit()
    conn.close()

    return jsonify(result)


@app.route('/api/history', methods=['GET'])
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT id, prediction, confidence, latitude, longitude, timestamp FROM detections').fetchall()
    conn.close()
    return jsonify(rows)


@app.route('/api/priority', methods=['GET'])
def priority():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT id, prediction, latitude, longitude FROM detections').fetchall()
    conn.close()

    rank = {'full': 0, 'partial': 1, 'empty': 2}
    sorted_rows = sorted(rows, key=lambda r: rank.get(r[1], 99))
    return jsonify(sorted_rows)


if __name__ == '__main__':
    app.run(debug=True)
