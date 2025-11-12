from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
    redirect,
    url_for,
)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from io import StringIO, BytesIO
from datetime import timedelta

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")
app.config["UPLOAD_FOLDER"] = "uploads"
app.permanent_session_lifetime = timedelta(hours=4)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# In-memory "storage" keyed by session id (very simple)
DATA_STORE = {}


# -----------------------
# Helpers
# -----------------------
def get_session_key():
    return session.get("sid")


def save_df_for_session(df):
    sid = get_session_key()
    if not sid:
        sid = os.urandom(16).hex()
        session["sid"] = sid
    DATA_STORE[sid] = df


def load_df_for_session():
    sid = get_session_key()
    if not sid:
        return None
    return DATA_STORE.get(sid)


def df_basic_stats(df):
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return {
        "rows": int(rows),
        "columns": int(cols),
        "numericColumns": len(numeric_cols),
        "numericCols": numeric_cols,
        "categoricalCols": categorical_cols,
    }


# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    # single page app (login + dashboard)
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if username and password:
        session.permanent = True
        session["user"] = username
        # create SID and empty store
        sid = os.urandom(16).hex()
        session["sid"] = sid
        DATA_STORE[sid] = None
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Missing credentials"}), 400


@app.route("/logout", methods=["POST"])
def logout():
    sid = session.pop("sid", None)
    session.pop("user", None)
    if sid and sid in DATA_STORE:
        DATA_STORE.pop(sid)
    return jsonify({"ok": True})


# File upload (multipart form)
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "No file selected"}), 400
    # accept CSV only
    filename = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(filename)
    # parse CSV with pandas
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to read CSV: {e}"}), 400
    save_df_for_session(df)
    stats = df_basic_stats(df)
    # return first 10 rows as JSON for table preview
    sample = df.head(10).fillna("").to_dict(orient="records")
    return jsonify(
        {
            "ok": True,
            "stats": stats,
            "sample": sample,
            "filename": os.path.basename(filename),
        }
    )


@app.route("/api/view", methods=["GET"])
def api_view():
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    stats = df_basic_stats(df)
    sample = df.head(10).fillna("").to_dict(orient="records")
    return jsonify({"ok": True, "stats": stats, "sample": sample})


# -----------------------
# Preprocessing / actions
# -----------------------
@app.route("/api/preprocess/missing", methods=["POST"])
def handle_missing():
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    processed = []
    total_missing = int(df.isna().sum().sum())

    # numeric: fill with mean
    for col in numeric_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            filled = int(df[col].isna().sum())
            df[col].fillna(round(mean_val, 4), inplace=True)
            processed.append(
                {
                    "column": col,
                    "type": "Numeric",
                    "method": "Mean",
                    "filled": filled,
                    "value": round(mean_val, 4),
                }
            )

    # categorical: fill with mode
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                mode_val = mode_val.iloc[0]
                filled = int(df[col].isna().sum())
                df[col].fillna(mode_val, inplace=True)
                processed.append(
                    {
                        "column": col,
                        "type": "Categorical",
                        "method": "Mode",
                        "filled": filled,
                        "value": str(mode_val),
                    }
                )

    save_df_for_session(df)
    return jsonify(
        {
            "ok": True,
            "totalMissing": total_missing,
            "processedCols": processed,
            "sample": df.head(5).fillna("").to_dict(orient="records"),
        }
    )


@app.route("/api/preprocess/normalize", methods=["POST"])
def normalize():
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min != 0:
            df[col] = ((df[col] - col_min) / (col_max - col_min)).round(6)
    save_df_for_session(df)
    return jsonify(
        {
            "ok": True,
            "message": "Data normalized",
            "sample": df.head(5).fillna("").to_dict(orient="records"),
        }
    )


@app.route("/api/preprocess/encode", methods=["POST"])
def encode_categorical():
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    df = df.copy()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    encodings = {}
    for col in cat_cols:
        le = LabelEncoder()
        # fillna placeholder to allow encoding
        df[col] = df[col].fillna("___MISSING___").astype(str)
        df[col + "_encoded"] = le.fit_transform(df[col])
        encodings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    save_df_for_session(df)
    return jsonify(
        {
            "ok": True,
            "encodedColumns": list(encodings.keys()),
            "encodings": encodings,
            "sample": df.head(5).to_dict(orient="records"),
        }
    )


@app.route("/api/preprocess/remove_outliers", methods=["POST"])
def remove_outliers():
    payload = request.json or {}
    method = payload.get("method", "iqr")  # support 'iqr' or 'zscore'
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    original_len = len(df)
    removed_count = 0

    if method == "iqr":
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask &= df[col].between(lower, upper, inclusive="both")
        df = df[mask]
        removed_count = original_len - len(df)
    else:  # zscore threshold 3
        from scipy import stats

        zscores = np.abs(stats.zscore(df[numeric_cols].dropna(), nan_policy="omit"))
        if zscores.size == 0:
            removed_count = 0
        else:
            if zscores.ndim == 1:
                mask = zscores <= 3
            else:
                mask = (zscores <= 3).all(axis=1)
            df = df.loc[df[numeric_cols].dropna().index[mask]]
            removed_count = original_len - len(df)

    save_df_for_session(df)
    return jsonify(
        {
            "ok": True,
            "removed": int(removed_count),
            "remainingRows": int(len(df)),
            "sample": df.head(5).fillna("").to_dict(orient="records"),
        }
    )


@app.route("/api/detect_outliers", methods=["GET"])
def detect_outliers():
    method = request.args.get("method", "iqr")
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = []
    for col in numeric_cols:
        col_vals = df[col].dropna().to_numpy()
        if len(col_vals) == 0:
            continue
        if method == "iqr":
            q1 = np.quantile(col_vals, 0.25)
            q3 = np.quantile(col_vals, 0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            for idx, v in df[col].dropna().items():
                if v < low or v > high:
                    outliers.append(
                        {
                            "column": col,
                            "index": int(idx),
                            "value": float(v),
                            "method": "IQR",
                        }
                    )
        else:
            mean = col_vals.mean()
            std = col_vals.std()
            if std == 0:
                continue
            for idx, v in df[col].dropna().items():
                z = abs((v - mean) / std)
                if z > 3:
                    outliers.append(
                        {
                            "column": col,
                            "index": int(idx),
                            "value": float(v),
                            "zscore": float(z),
                            "method": "Z-Score",
                        }
                    )
    return jsonify({"ok": True, "outliers": outliers, "count": len(outliers)})


# -----------------------
# Visualization & Report endpoints
# -----------------------
from math import isfinite
from html import escape


def _numeric_series(df, col):
    # return numeric values (dropna) as python list of floats
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s


@app.route("/api/visual/univariate", methods=["GET"])
def visual_univariate():
    cols_param = request.args.get("cols")
    bins = int(request.args.get("bins", 30))
    if not cols_param:
        return (
            jsonify(
                {"ok": False, "error": "cols parameter required (comma separated)"}
            ),
            400,
        )
    cols = [c.strip() for c in cols_param.split(",") if c.strip()]
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    out = {}
    for col in cols:
        if col not in df.columns:
            out[col] = {"error": "column not found"}
            continue
        s = _numeric_series(df, col)
        if s is None or len(s) == 0:
            out[col] = {"count": 0, "values": [], "summary": {}}
            continue
        vals = s.tolist()
        summary = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "missing": int(df[col].isna().sum()),
        }
        out[col] = {
            "count": summary["count"],
            "values": vals,
            "summary": summary,
            "bins": bins,
        }
    return jsonify({"ok": True, "data": out})


@app.route("/api/visual/bivariate", methods=["GET"])
def visual_bivariate():
    xcol = request.args.get("x")
    ycol = request.args.get("y")
    if not xcol or not ycol:
        return jsonify({"ok": False, "error": "x and y parameters required"}), 400
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    if xcol not in df.columns or ycol not in df.columns:
        return jsonify({"ok": False, "error": "columns not found"}), 400
    # coerce to numeric and drop NaNs jointly
    sub = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    x = sub[xcol].tolist()
    y = sub[ycol].tolist()
    # quick linear fit (optional)
    corr = None
    try:
        if len(x) > 1:
            arrx = np.array(x)
            arry = np.array(y)
            corr = float(np.corrcoef(arrx, arry)[0, 1])
    except Exception:
        corr = None
    return jsonify({"ok": True, "x": x, "y": y, "count": len(x), "corr": corr})


@app.route("/api/visual/multivariate", methods=["GET"])
def visual_multivariate():
    limit = int(request.args.get("limit", 10))
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return jsonify({"ok": False, "error": "No numeric columns available"}), 400
    # limit number of columns for performance (choose top variance columns)
    variances = numeric.var().sort_values(ascending=False)
    chosen = variances.index[:limit].tolist()
    corr = numeric[chosen].corr().round(4)
    labels = chosen
    matrix = corr.values.tolist()
    return jsonify({"ok": True, "labels": labels, "matrix": matrix})


@app.route("/api/report", methods=["GET"])
def generate_report_api():
    """
    Returns a JSON report and an HTML snippet for quick rendering.
    """
    df = load_df_for_session()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400

    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # missing value summary
    missing = df.isna().sum().to_dict()
    missing_summary = [
        {"column": k, "missing": int(v)} for k, v in missing.items() if v > 0
    ]
    missing_summary = sorted(missing_summary, key=lambda x: -x["missing"])

    # top correlations (absolute)
    corr_report = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs().fillna(0)
        # find top pairs
        pairs = []
        n = len(numeric_cols)
        for i in range(n):
            for j in range(i + 1, n):
                a = numeric_cols[i]
                b = numeric_cols[j]
                pairs.append((a, b, float(corr_matrix.loc[a, b])))
        pairs_sorted = sorted(pairs, key=lambda x: -x[2])[:10]
        corr_report = [
            {"col1": a, "col2": b, "abs_corr": c} for a, b, c in pairs_sorted
        ]

    # basic statistics for numeric columns
    num_stats = {}
    for c in numeric_cols:
        s = _numeric_series(df, c)
        if s is None or len(s) == 0:
            continue
        num_stats[c] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.median()),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    json_report = {
        "ok": True,
        "summary": {"rows": int(rows), "columns": int(cols)},
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_summary": missing_summary,
        "top_correlations": corr_report,
        "numeric_stats": num_stats,
    }

    # create a small HTML snippet (safe-escaped)
    html_parts = []
    html_parts.append(f"<h3>Dataset Report</h3>")
    html_parts.append(
        f"<p><strong>Rows:</strong> {rows} &nbsp; <strong>Columns:</strong> {cols}</p>"
    )
    if missing_summary:
        html_parts.append("<h4>Missing Values</h4><ul>")
        for m in missing_summary[:10]:
            html_parts.append(f"<li>{escape(m['column'])}: {m['missing']} missing</li>")
        html_parts.append("</ul>")
    if corr_report:
        html_parts.append("<h4>Top correlations (absolute)</h4><ol>")
        for item in corr_report[:10]:
            html_parts.append(
                f"<li>{escape(item['col1'])} &amp; {escape(item['col2'])} — {item['abs_corr']:.3f}</li>"
            )
        html_parts.append("</ol>")
    if numeric_cols:
        html_parts.append("<h4>Numeric column summary (first 10)</h4><ul>")
        for c in numeric_cols[:10]:
            st = num_stats.get(c)
            if st:
                html_parts.append(
                    f"<li>{escape(c)} — mean: {st['mean']:.3f}, std: {st['std']:.3f}, min: {st['min']}, max: {st['max']}</li>"
                )
        html_parts.append("</ul>")

    html_snippet = "\n".join(html_parts)
    json_report["html"] = html_snippet
    return jsonify(json_report)


# -----------------------
# Export CSV
# -----------------------
# -----------------------
# Lightweight ML (no scikit-learn) - pure numpy implementations
# -----------------------
import math
import pickle
from typing import Dict, Any


# Simple train/test split replacement (numpy)
def simple_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42
):
    rng = np.random.RandomState(int(random_state))
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# Model classes (lightweight)
class NumpyLinearRegression:
    def __init__(self):
        self.coef_ = None  # including intercept at index 0 if fit_intercept=True
        self.intercept_ = 0.0
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Add intercept column
        ones = np.ones((X.shape[0], 1))
        A = np.hstack([ones, X])  # shape n x (m+1)
        # normal equation with pseudo-inverse
        w = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(y)
        self.intercept_ = float(w[0])
        self.coef_ = w[1:].astype(float)
        return self

    def predict(self, X: np.ndarray):
        return (np.dot(X, self.coef_) + self.intercept_).astype(float)


class NumpyLogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000, tol=1e-6):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, m = X.shape
        self.w = np.zeros(m, dtype=float)
        self.b = 0.0
        prev_loss = None
        for it in range(self.n_iter):
            z = X.dot(self.w) + self.b
            preds = self._sigmoid(z)
            # binary cross-entropy gradient
            error = preds - y
            grad_w = (X.T @ error) / n
            grad_b = np.mean(error)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            # optional stopping
            loss = -np.mean(
                y * np.log(preds + 1e-12) + (1 - y) * np.log(1 - preds + 1e-12)
            )
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X: np.ndarray):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X: np.ndarray, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class NumpyKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.centers = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n, m = X.shape
        # initialize centers as random samples
        initial_idx = rng.choice(n, self.n_clusters, replace=False)
        centers = X[initial_idx].astype(float)
        labels = np.zeros(n, dtype=int)
        for it in range(self.max_iter):
            # assign clusters
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # n x k
            new_labels = np.argmin(dists, axis=1)
            # update centers
            new_centers = np.array(
                [
                    (
                        X[new_labels == k].mean(axis=0)
                        if np.any(new_labels == k)
                        else centers[k]
                    )
                    for k in range(self.n_clusters)
                ]
            )
            move = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labels = new_labels
            if move <= self.tol:
                break
        self.centers = centers
        self.labels_ = labels
        self.inertia_ = float(
            np.sum((np.linalg.norm(X - centers[labels], axis=1)) ** 2)
        )
        return self

    def predict(self, X: np.ndarray):
        dists = np.linalg.norm(X[:, None, :] - self.centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


class NumpyPCA:
    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        # center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD
        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
        components = VT[: self.n_components]
        explained = (S**2) / (X.shape[0] - 1)
        total_var = explained.sum()
        self.components_ = components
        self.explained_variance_ratio_ = (
            explained[: self.n_components] / total_var
        ).tolist()
        return self

    def transform(self, X: np.ndarray):
        Xc = X - self.mean_
        return np.dot(Xc, self.components_.T)


# Helpers to persist models per-session inside DATA_STORE (model meta saved under 'models' key)
def save_model_for_session(name: str, model_obj: Any, features, target=None, meta=None):
    sid = get_session_key()
    if not sid:
        return False
    session_store = DATA_STORE.get(sid, {})
    if not isinstance(session_store, dict):
        session_store = {}
    models = session_store.get("models", {})
    models[name] = {
        "model": pickle.dumps(model_obj),
        "features": features,
        "target": target,
        "meta": meta or {},
    }
    session_store["models"] = models
    if sid in DATA_STORE and isinstance(DATA_STORE[sid], pd.DataFrame):
        session_store["df"] = DATA_STORE[sid]
    DATA_STORE[sid] = session_store
    return True


def load_model_for_session(name: str):
    sid = get_session_key()
    if not sid:
        return None
    session_store = DATA_STORE.get(sid, {})
    models = session_store.get("models", {})
    info = models.get(name)
    if not info:
        return None
    try:
        model_obj = pickle.loads(info["model"])
    except Exception:
        return None
    return {
        "model": model_obj,
        "features": info.get("features"),
        "target": info.get("target"),
        "meta": info.get("meta"),
    }


def list_models_for_session():
    sid = get_session_key()
    if not sid:
        return []
    session_store = DATA_STORE.get(sid, {})
    models = session_store.get("models", {})
    return [
        {
            "name": k,
            "features": v.get("features"),
            "target": v.get("target"),
            "meta": v.get("meta"),
        }
        for k, v in models.items()
    ]


# API: train model
@app.route("/api/models/train", methods=["POST"])
def train_model():
    payload = request.json or {}
    model_name = payload.get("name") or f"model_{os.urandom(4).hex()}"
    model_type = payload.get("model_type")
    features = payload.get("features")
    target = payload.get("target")
    test_size = float(payload.get("test_size", 0.2))
    random_state = int(payload.get("random_state", 42))
    params = payload.get("params", {})

    # get session dataframe (compatibility)
    df_store = load_df_for_session()
    if isinstance(df_store, dict) and "df" in df_store:
        df = df_store["df"]
    else:
        df = df_store

    if df is None or not isinstance(df, pd.DataFrame):
        return jsonify({"ok": False, "error": "No dataset loaded"}), 400

    if model_type not in ["linear_regression", "logistic_regression", "kmeans", "pca"]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Unsupported model_type (supported: linear_regression, logistic_regression, kmeans, pca)",
                }
            ),
            400,
        )

    try:
        df_copy = df.copy()
        if not features:
            return jsonify({"ok": False, "error": "features required"}), 400

        # prepare numeric X
        X_df = df_copy[features].apply(pd.to_numeric, errors="coerce")
        if model_type in ["linear_regression", "logistic_regression"]:
            if not target:
                return (
                    jsonify(
                        {"ok": False, "error": "target required for supervised models"}
                    ),
                    400,
                )
            y_series = df_copy[target]
            # align rows
            combined = pd.concat([X_df, y_series], axis=1).dropna()
            X_df = combined[features]
            y_series = combined[target]
            # convert y for logistic if necessary (binary)
            if model_type == "logistic_regression":
                # factorize to 0/1 if needed
                if y_series.dtype == object or len(np.unique(y_series)) != 2:
                    y_series = pd.factorize(y_series)[0]
                else:
                    y_series = y_series.astype(int).values
        else:
            # clustering / PCA: drop NA rows
            combined = X_df.dropna()
            X_df = combined

        X = X_df.to_numpy(dtype=float)

        # train/test split
        if model_type in ["linear_regression", "logistic_regression"]:
            X_train_df, X_test_df, y_train_series, y_test_series = (
                simple_train_test_split(
                    X_df.reset_index(drop=True),
                    pd.Series(y_series).reset_index(drop=True),
                    test_size=test_size,
                    random_state=random_state,
                )
            )
            X_train = X_train_df.to_numpy(dtype=float)
            X_test = X_test_df.to_numpy(dtype=float)
            y_train = np.array(y_train_series)
            y_test = np.array(y_test_series)
        else:
            X_train = X
            X_test = None
            y_train = None
            y_test = None

        # instantiate and fit
        if model_type == "linear_regression":
            model = NumpyLinearRegression()
            model.fit(X_train, y_train)
        elif model_type == "logistic_regression":
            lr = float(params.get("lr", 0.1))
            n_iter = int(params.get("n_iter", 1000))
            model = NumpyLogisticRegression(lr=lr, n_iter=n_iter)
            model.fit(X_train, y_train)
        elif model_type == "kmeans":
            n_clusters = int(params.get("n_clusters", 3))
            model = NumpyKMeans(n_clusters=n_clusters, random_state=random_state)
            model.fit(X_train)
        elif model_type == "pca":
            n_components = int(params.get("n_components", min(X_train.shape[1], 2)))
            model = NumpyPCA(n_components=n_components)
            model.fit(X_train)

        # quick evaluation
        eval_result = {}
        if model_type == "linear_regression":
            preds = model.predict(X_test)
            eval_result = {
                "mse": float(np.mean((y_test - preds) ** 2)),
                "mae": float(np.mean(np.abs(y_test - preds))),
                "r2": (
                    float(
                        1
                        - (
                            np.sum((y_test - preds) ** 2)
                            / np.sum((y_test - np.mean(y_test)) ** 2)
                        )
                    )
                    if len(y_test) > 0
                    else 0.0
                ),
            }
        elif model_type == "logistic_regression":
            preds = model.predict(X_test)
            y_true = y_test.astype(int)
            acc = float((preds == y_true).mean()) if len(y_true) > 0 else 0.0
            # simple precision/recall/f1 for binary
            tp = int(((preds == 1) & (y_true == 1)).sum()) if len(y_true) > 0 else 0
            fp = int(((preds == 1) & (y_true == 0)).sum()) if len(y_true) > 0 else 0
            fn = int(((preds == 0) & (y_true == 1)).sum()) if len(y_true) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            eval_result = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        elif model_type == "kmeans":
            eval_result = {"inertia": float(model.inertia_)}
        elif model_type == "pca":
            eval_result = {
                "explained_variance_ratio": [
                    float(x) for x in model.explained_variance_ratio_
                ]
            }

        meta = {
            "model_type": model_type,
            "trained_on_rows": int(X_train.shape[0]),
            "features": features,
            "target": target,
        }
        save_model_for_session(model_name, model, features, target=target, meta=meta)
        return jsonify(
            {"ok": True, "model": model_name, "eval": eval_result, "meta": meta}
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"Training failed: {e}"}), 500


# list models
@app.route("/api/models/list", methods=["GET"])
def models_list():
    models = list_models_for_session()
    return jsonify({"ok": True, "models": models})


# predict
@app.route("/api/models/predict", methods=["POST"])
def models_predict():
    payload = request.json or {}
    model_name = payload.get("model")
    data = payload.get("data")
    if not model_name or not data:
        return jsonify({"ok": False, "error": "model and data required"}), 400
    info = load_model_for_session(model_name)
    if not info:
        return jsonify({"ok": False, "error": "model not found"}), 404
    model = info["model"]
    features = info["features"]
    try:
        df_in = pd.DataFrame(data)
        X_df = df_in[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        X = X_df.to_numpy(dtype=float)
        preds = model.predict(X)
        preds_out = []
        for p in preds:
            if isinstance(p, (np.generic, np.ndarray)):
                p = p.tolist()
            preds_out.append(p)
        return jsonify({"ok": True, "predictions": preds_out})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Predict failed: {e}"}), 500


# evaluate
@app.route("/api/models/evaluate", methods=["POST"])
def models_evaluate():
    payload = request.json or {}
    model_name = payload.get("model")
    X_list = payload.get("X")
    y_list = payload.get("y")
    test_size = float(payload.get("test_size", 0.2))
    info = load_model_for_session(model_name)
    if not info:
        return jsonify({"ok": False, "error": "model not found"}), 404
    model = info["model"]
    features = info["features"]
    target = info["target"]
    # load session df
    df_store = load_df_for_session()
    if isinstance(df_store, dict) and "df" in df_store:
        df = df_store["df"]
    else:
        df = df_store
    try:
        if X_list and y_list:
            X_df = pd.DataFrame(X_list)[features].apply(pd.to_numeric, errors="coerce")
            y_series = pd.Series(y_list)
            combined = pd.concat([X_df, y_series], axis=1).dropna()
            X = combined[features].to_numpy(dtype=float)
            y = combined.iloc[:, -1].to_numpy()
        else:
            if df is None or target is None:
                return jsonify({"ok": False, "error": "No test data available"}), 400
            X_df = df[features].apply(pd.to_numeric, errors="coerce")
            y_series = df[target]
            combined = pd.concat([X_df, y_series], axis=1).dropna()
            X_all = combined[features]
            y_all = combined[target]
            X_train_df, X_test_df, y_train_series, y_test_series = (
                simple_train_test_split(
                    X_all.reset_index(drop=True),
                    pd.Series(y_all).reset_index(drop=True),
                    test_size=test_size,
                    random_state=42,
                )
            )
            X = X_test_df.to_numpy(dtype=float)
            y = np.array(y_test_series)

        preds = model.predict(X)
        mtype = info["meta"].get("model_type")
        if mtype == "linear_regression":
            res = {
                "mse": float(np.mean((y - preds) ** 2)),
                "mae": float(np.mean(np.abs(y - preds))),
                "r2": (
                    float(
                        1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    )
                    if len(y) > 0
                    else 0.0
                ),
            }
        elif mtype == "logistic_regression":
            y_true = y.astype(int)
            acc = float((preds == y_true).mean()) if len(y_true) > 0 else 0.0
            tp = int(((preds == 1) & (y_true == 1)).sum()) if len(y_true) > 0 else 0
            fp = int(((preds == 1) & (y_true == 0)).sum()) if len(y_true) > 0 else 0
            fn = int(((preds == 0) & (y_true == 1)).sum()) if len(y_true) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            res = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
        else:
            res = {
                "message": "Evaluation not defined for this model type",
                "n": int(len(preds)),
            }
        return jsonify({"ok": True, "metrics": res})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Evaluation failed: {e}"}), 500


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
