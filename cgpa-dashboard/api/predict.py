# Prediksi CGPA tanpa scikit-learn/pandas/numpy: model Random Forest dibaca dari model.json
# (hasil ekspor best_model.pkl), sehingga fungsi Vercel kecil dan tidak bergantung versi library.
from http.server import BaseHTTPRequestHandler
import json, math, os, struct

_M = None

def get_model():
    global _M
    if _M is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model.json")
        with open(path) as f:
            _M = json.load(f)
    return _M

def f32(x):
    # sklearn membandingkan fitur dalam float32; ditiru agar hasil identik
    return struct.unpack("f", struct.pack("f", x))[0]

def predict(row):
    M = get_model()
    x = []
    for c in M["log"]:
        x.append((math.log1p(row[c["col"]]) - c["mean"]) / c["scale"])
    for c in M["plain"]:
        x.append((row[c["col"]] - c["mean"]) / c["scale"])
    for c in M["cat"]:
        for k in c["keep"]:
            x.append(1.0 if row[c["col"]] == k else 0.0)
    x = [f32(v) for v in x]
    total = 0.0
    for t in M["trees"]:
        n, left, feat, thr = 0, t["l"], t["f"], t["t"]
        while left[n] != -1:
            n = left[n] if x[feat[n]] <= thr[n] else t["r"][n]
        total += t["v"][n]
    return total / len(M["trees"])

YN = ["Yes", "No"]
# key dari frontend -> (nama kolom di dataset, tipe, batas / pilihan valid)
FIELDS = {
    "study_hours":     ("How many hour do you study daily?", "num", (0, 12)),
    "study_sessions":  ("How many times do you seat for study in a day?", "num", (0, 10)),
    "social_media":    ("How many hour do you spent daily in social media?", "num", (0, 10)),
    "skill_dev":       ("How many hour do you spent daily on your skill development?", "num", (0, 8)),
    "attendance":      ("Average attendance on class", "num", (0, 100)),
    "learning_mode":   ("What is your preferable learning mode?", "cat", ["Offline", "Online"]),
    "probation":       ("Did you ever fall in probation?", "cat", YN),
    "teacher_consult": ("Do you attend in teacher consultancy for any kind of academical problems?", "cat", YN),
    "co_curriculum":   ("Are you engaged with any co-curriculum activities?", "cat", YN),
    "gender":          ("Gender", "cat", ["Male", "Female"]),
    "age":             ("Age", "num", (17, 35)),
    "semester":        ("Current Semester", "num", (1, 8)),
    "scholarship":     ("Do you have meritorious scholarship ?", "cat", YN),
    "english_prof":    ("Status of your English language proficiency", "cat", ["Basic", "Intermediate", "Advance"]),
    "living_with":     ("With whom you are living with?", "cat", ["Family", "Bachelor"]),
    "health_issue":    ("Do you have any health issues?", "cat", YN),
    "family_income":   ("What is your monthly family income?", "num", (0, 50000000)),
    "personal_pc":     ("Do you have personal Computer?", "cat", YN),
    "relationship":    ("What is your relationship status?", "cat", ["Single", "In a relationship", "Engaged", "Married"]),
    "smartphone":      ("Do you use smart phone?", "cat", YN),
    "disability":      ("Do you have any physical disabilities?", "cat", YN),
    "transportation":  ("Do you use University transportation?", "cat", YN),
}

def build_row(data):
    row = {}
    for key, (col, kind, rule) in FIELDS.items():
        if key not in data:
            raise ValueError(f"Field '{key}' belum diisi")
        val = data[key]
        if kind == "num":
            val = min(max(float(val), rule[0]), rule[1])
        elif val not in rule:
            raise ValueError(f"Nilai '{val}' tidak valid untuk '{key}'")
        row[col] = val
    return row

class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(size) or b"{}")
            pred = predict(build_row(data))
            self._send(200, {"cgpa": round(min(4.0, max(0.0, pred)), 2)})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"Model gagal memprediksi: {e}"})
