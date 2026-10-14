from http.server import BaseHTTPRequestHandler
import json, os, pickle
import pandas as pd

_model = None

def get_model():
    global _model
    if _model is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_model.pkl")
        with open(path, "rb") as f:
            _model = pickle.load(f)
    return _model

YN = ["Yes", "No"]
# key dari frontend -> (nama kolom di dataset, tipe, batas / pilihan valid)
FIELDS = {
    "study_hours":     ("How many hour do you study daily?", "num", (0, 12)),
    "study_sessions":  ("How many times do you seat for study in a day?", "num", (0, 10)),
    "social_media":    ("How many hour do you spent daily in social media?", "num", (0, 10)),
    "skill_dev":       ("How many hour do you spent daily on your skill development?", "num", (0, 8)),
    "attendance":      ("Average attendance on class", "num", (0, 100)),
    "learning_mode":   ("What is your preferable learning mode?", "cat", ["Online", "Offline", "Blended"]),
    "probation":       ("Did you ever fall in probation?", "cat", YN),
    "teacher_consult": ("Do you attend in teacher consultancy for any kind of academical problems?", "cat", YN),
    "co_curriculum":   ("Are you engaged with any co-curriculum activities?", "cat", YN),
    "gender":          ("Gender", "cat", ["Male", "Female"]),
    "age":             ("Age", "num", (17, 35)),
    "semester":        ("Current Semester", "num", (1, 8)),
    "scholarship":     ("Do you have meritorious scholarship ?", "cat", YN),
    "english_prof":    ("Status of your English language proficiency", "cat", ["Basic", "Intermediate", "Advanced"]),
    "living_with":     ("With whom you are living with?", "cat", ["Family", "Friends", "Alone"]),
    "health_issue":    ("Do you have any health issues?", "cat", YN),
    "family_income":   ("What is your monthly family income?", "num", (0, 50000000)),
    "personal_pc":     ("Do you have personal Computer?", "cat", YN),
    "relationship":    ("What is your relationship status?", "cat", ["Single", "In Relationship"]),
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
            val = float(val)
            val = min(max(val, rule[0]), rule[1])
        elif val not in rule:
            raise ValueError(f"Nilai '{val}' tidak valid untuk '{key}'")
        row[col] = val
    return pd.DataFrame([row])

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
            pred = float(get_model().predict(build_row(data))[0])
            self._send(200, {"cgpa": round(min(4.0, max(0.0, pred)), 2)})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"Model gagal memprediksi: {e}"})
