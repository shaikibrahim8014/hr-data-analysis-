"""
Dataset Explorer Web App — Single-file Flask application
Features:
- Login page (simple username/password)
- Upload CSV/Excel dataset
- Dashboard with buttons: View Data, Missing Data, Outliers (Z-score & IQR), Visualizations (uni/bi/multi), Export Data, Generate Report
- Outlier detection: Z-score and IQR
- Visualizations: hist / bar / box / scatter for uni/bi/multi (Plotly for interactive plots)
- Exports: cleaned CSV, Excel, and downloadable HTML report

Requirements:
pip install flask pandas plotly openpyxl scipy python-magic

Run:
python dataset_web_app.py
Open http://127.0.0.1:5000

Note: This is a minimal, single-file demo suitable for local use and further enhancement.
"""
from flask import Flask, request, redirect, url_for, render_template_string, session, send_file, flash
import pandas as pd
import io
import os
import base64
import json
from scipy import stats
import plotly.express as px
import plotly.io as pio
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(hours=5)

# ---------- Simple in-memory storage for demo ----------
# In production use a DB or persistent storage and secure auth
DATASTORE = {
    'df': None,        # current uploaded dataframe
    'filename': None,  # original filename
}

# ---------- Templates (kept inline so app is single-file) ----------
login_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Dataset Explorer — Login</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center">
  <div class="max-w-md w-full bg-white p-8 rounded-2xl shadow-lg">
    <h1 class="text-2xl font-semibold mb-4">Dataset Explorer</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="mb-3 text-sm text-red-600">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="post" action="{{ url_for('login') }}">
      <label class="block text-sm">Username</label>
      <input class="w-full border rounded px-3 py-2 mb-3" name="username" required>
      <label class="block text-sm">Password</label>
      <input type="password" class="w-full border rounded px-3 py-2 mb-4" name="password" required>
      <button class="w-full bg-indigo-600 text-white py-2 rounded hover:opacity-90">Login</button>
    </form>
    <p class="mt-4 text-xs text-slate-500">Use username: <code>user</code> and password: <code>pass</code> for demo</p>
  </div>
</body>
</html>
'''

upload_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Upload Dataset</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 to-white min-h-screen p-8">
  <div class="max-w-4xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-6">
      <h2 class="text-xl font-semibold">Upload dataset</h2>
      <div>
        <a href="{{ url_for('dashboard') }}" class="text-sm text-indigo-600">Dashboard</a>
        <a href="{{ url_for('logout') }}" class="ml-4 text-sm text-red-600">Logout</a>
      </div>
    </div>

    <form action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data" class="space-y-4">
      <input type="file" name="datafile" accept=".csv,.xls,.xlsx" required>
      <div>
        <button class="bg-indigo-600 text-white px-4 py-2 rounded">Upload</button>
      </div>
    </form>

    {% if filename %}
    <div class="mt-6 p-4 bg-slate-50 rounded">
      <div class="text-sm text-slate-600">Current file: <strong>{{ filename }}</strong></div>
      <div class="mt-2 text-sm">
        <a href="{{ url_for('dashboard') }}" class="text-indigo-600">Go to dashboard</a>
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
'''

# Dashboard with buttons
dashboard_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Dataset Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 to-white min-h-screen p-8">
  <div class="max-w-5xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-6">
      <h2 class="text-2xl font-semibold">Dashboard — {{ filename if filename else 'no file uploaded' }}</h2>
      <div>
        <a href="{{ url_for('upload') }}" class="text-sm text-indigo-600">Upload new file</a>
        <a href="{{ url_for('logout') }}" class="ml-4 text-sm text-red-600">Logout</a>
      </div>
    </div>

    {% if not df_exists %}
      <div class="p-6 text-center text-slate-600">No dataset uploaded yet. Please <a href="{{ url_for('upload') }}" class="text-indigo-600">upload a dataset</a>.</div>
    {% else %}
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <a href="{{ url_for('view_data') }}" class="block p-4 bg-indigo-50 rounded hover:shadow">View Data</a>
      <a href="{{ url_for('missing_data') }}" class="block p-4 bg-amber-50 rounded hover:shadow">Missing Data</a>
      <a href="{{ url_for('outliers') }}" class="block p-4 bg-rose-50 rounded hover:shadow">Outliers</a>
      <a href="{{ url_for('visualize') }}" class="block p-4 bg-emerald-50 rounded hover:shadow">Visualize</a>
      <a href="{{ url_for('export_data') }}" class="block p-4 bg-slate-50 rounded hover:shadow">Export Data</a>
      <a href="{{ url_for('generate_report') }}" class="block p-4 bg-yellow-50 rounded hover:shadow">Generate Report</a>
    </div>
    {% endif %}
  </div>
</body>
</html>
'''

# View data template
view_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>View Data</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 p-8">
  <div class="max-w-6xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold">Data preview — {{ filename }}</h3>
      <div>
        <a href="{{ url_for('dashboard') }}" class="text-sm text-indigo-600">Back to dashboard</a>
      </div>
    </div>
    <div class="overflow-x-auto">
      {{ table|safe }}
    </div>
  </div>
</body>
</html>
'''

missing_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Missing Data</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 p-8">
  <div class="max-w-4xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold">Missing data report</h3>
      <div>
        <a href="{{ url_for('dashboard') }}" class="text-sm text-indigo-600">Back to dashboard</a>
      </div>
    </div>

    <div class="space-y-4">
      <div class="p-4 bg-amber-50 rounded">
        <div class="text-sm text-slate-700">Total rows: <strong>{{ total_rows }}</strong></div>
        <div class="text-sm text-slate-700">Columns with missing values: <strong>{{ missing_cols|length }}</strong></div>
      </div>

      <div class="overflow-x-auto">
        {{ table|safe }}
      </div>

      <div class="p-4 bg-slate-50 rounded">
        <form action="{{ url_for('missing_data') }}" method="post">
          <label class="text-sm">Fill missing with:</label>
          <select name="method" class="ml-2 border rounded px-2 py-1">
            <option value="drop">Drop rows with any missing</option>
            <option value="mean">Numeric: fill with mean</option>
            <option value="median">Numeric: fill with median</option>
            <option value="mode">Fill with mode (per column)</option>
          </select>
          <button class="ml-3 bg-indigo-600 text-white px-3 py-1 rounded">Apply</button>
        </form>
      </div>

    </div>
  </div>
</body>
</html>
'''

outliers_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Outliers</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 p-8">
  <div class="max-w-5xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold">Outlier analysis</h3>
      <div>
        <a href="{{ url_for('dashboard') }}" class="text-sm text-indigo-600">Back to dashboard</a>
      </div>
    </div>

    <form method="get" class="mb-4">
      <label class="text-sm">Method:</label>
      <select name="method" class="ml-2 border rounded px-2 py-1">
        <option value="zscore" {% if method=='zscore' %}selected{% endif %}>Z-score (threshold 3)</option>
        <option value="iqr" {% if method=='iqr' %}selected{% endif %}>IQR (1.5 * IQR)</option>
      </select>
      <label class="ml-4 text-sm">Column:</label>
      <select name="col" class="ml-2 border rounded px-2 py-1">
        <option value="_all_">All numeric columns</option>
        {% for c in numeric_cols %}
          <option value="{{ c }}" {% if selected_col==c %}selected{% endif %}>{{ c }}</option>
        {% endfor %}
      </select>
      <button class="ml-4 bg-indigo-600 text-white px-3 py-1 rounded">Run</button>
    </form>

    <div class="space-y-4">
      <div class="p-3 bg-slate-50 rounded">Outlier count: <strong>{{ outlier_count }}</strong></div>
      <div class="overflow-x-auto">{{ table|safe }}</div>
    </div>
  </div>
</body>
</html>
'''

visualize_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Visualize</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 p-8">
  <div class="max-w-6xl mx-auto bg-white p-6 rounded-2xl shadow">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold">Visualize</h3>
      <div>
        <a href="{{ url_for('dashboard') }}" class="text-sm text-indigo-600">Back to dashboard</a>
      </div>
    </div>

    <form method="get" class="space-y-3 mb-4">
      <div>
        <label class="text-sm">Type:</label>
        <select name="vtype" class="ml-2 border rounded px-2 py-1">
          <option value="uni" {% if vtype=='uni' %}selected{% endif %}>Univariate</option>
          <option value="bi" {% if vtype=='bi' %}selected{% endif %}>Bivariate</option>
          <option value="multi" {% if vtype=='multi' %}selected{% endif %}>Multivariate</option>
        </select>
      </div>

      <div>
        <label class="text-sm">X column:</label>
        <select name="xcol" class="ml-2 border rounded px-2 py-1">
          {% for c in cols %}
            <option value="{{ c }}" {% if xcol==c %}selected{% endif %}>{{ c }}</option>
          {% endfor %}
        </select>

        <label class="ml-4 text-sm">Y column (for bi/multi):</label>
        <select name="ycol" class="ml-2 border rounded px-2 py-1">
          <option value="">(none)</option>
          {% for c in cols %}
            <option value="{{ c }}" {% if ycol==c %}selected{% endif %}>{{ c }}</option>
          {% endfor %}

        </select>

        <label class="ml-4 text-sm">Plot:</label>
        <select name="kind" class="ml-2 border rounded px-2 py-1">
          <option value="hist" {% if kind=='hist' %}selected{% endif %}>Histogram</option>
          <option value="box" {% if kind=='box' %}selected{% endif %}>Box</option>
          <option value="bar" {% if kind=='bar' %}selected{% endif %}>Bar</option>
          <option value="scatter" {% if kind=='scatter' %}selected{% endif %}>Scatter</option>
        </select>

        <button class="ml-4 bg-indigo-600 text-white px-3 py-1 rounded">Plot</button>
      </div>
    </form>

    <div>
      {% if plot_html %}
        <div>{{ plot_html|safe }}</div>
      {% else %}
        <div class="p-6 text-slate-500">Choose columns and press "Plot" to generate a visualization.</div>
      {% endif %}
    </div>
  </div>
</body>
</html>
'''

report_tmpl = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Generated Report</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="p-8 bg-white text-slate-800">
  <div class="max-w-5xl mx-auto">
    <h1 class="text-2xl font-semibold mb-2">Dataset Report: {{ filename }}</h1>
    <div class="text-sm text-slate-600 mb-6">Generated at {{ ts }}</div>

    <h2 class="text-lg font-semibold mt-4">Basic summary</h2>
    <div class="mt-2">{{ summary_html|safe }}</div>

    <h2 class="text-lg font-semibold mt-4">Missing values</h2>
    <div class="mt-2">{{ missing_html|safe }}</div>

    <h2 class="text-lg font-semibold mt-4">Outliers (brief)</h2>
    <div class="mt-2">{{ outliers_html|safe }}</div>

    <h2 class="text-lg font-semibold mt-4">Sample visual</h2>
    <div class="mt-2">{{ plot_html|safe }}</div>
  </div>
</body>
</html>
'''

# ---------- Utility helpers ----------

def allowed_file(filename):
    return filename.lower().endswith(('.csv', '.xls', '.xlsx'))


def read_file_to_df(file_storage):
    filename = file_storage.filename
    if filename.lower().endswith('.csv'):
        return pd.read_csv(file_storage)
    else:
        return pd.read_excel(file_storage)


def df_to_html_table(df, maxrows=100):
    return df.head(maxrows).to_html(classes='min-w-full text-sm', index=False, border=0)


def detect_outliers_zscore(df, col, thresh=3.0):
    series = df[col].dropna()
    if series.empty:
        return pd.Series(dtype=bool)
    z = (series - series.mean())/series.std(ddof=0)
    return (z.abs() > thresh).reindex(df.index, fill_value=False)


def detect_outliers_iqr(df, col, factor=1.5):
    series = df[col]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return ((series < low) | (series > high)).fillna(False)

# ---------- Routes ----------

@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # DEMO-only credentials
        if username == 'user' and password == 'pass':
            session['logged_in'] = True
            session.permanent = True
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials — demo expects user / pass')
            return render_template_string(login_tmpl)
    return render_template_string(login_tmpl)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        f = request.files.get('datafile')
        if not f or not allowed_file(f.filename):
            flash('Please upload a CSV or Excel file')
            return redirect(request.url)
        try:
            df = read_file_to_df(f)
        except Exception as e:
            flash('Failed to read file: ' + str(e))
            return redirect(request.url)
        DATASTORE['df'] = df
        DATASTORE['filename'] = f.filename
        return redirect(url_for('dashboard'))
    return render_template_string(upload_tmpl, filename=DATASTORE.get('filename'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df_exists = DATASTORE['df'] is not None
    return render_template_string(dashboard_tmpl, df_exists=df_exists, filename=DATASTORE.get('filename'))

@app.route('/view')
def view_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))
    table = df_to_html_table(df, maxrows=200)
    return render_template_string(view_tmpl, table=table, filename=DATASTORE.get('filename'))

@app.route('/missing', methods=['GET', 'POST'])
def missing_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))
    total_rows = len(df)
    missing = df.isnull().sum()
    missing = missing[missing>0].sort_values(ascending=False)
    table = missing.to_frame(name='missing_count').to_html(classes='min-w-full text-sm', border=0)

    if request.method == 'POST':
        method = request.form.get('method')
        if method == 'drop':
            DATASTORE['df'] = df.dropna()
        elif method == 'mean':
            for c in df.select_dtypes(include='number').columns:
                df[c] = df[c].fillna(df[c].mean())
            DATASTORE['df'] = df
        elif method == 'median':
            for c in df.select_dtypes(include='number').columns:
                df[c] = df[c].fillna(df[c].median())
            DATASTORE['df'] = df
        elif method == 'mode':
            for c in df.columns:
                if df[c].isnull().any():
                    mode = df[c].mode()
                    if not mode.empty:
                        df[c] = df[c].fillna(mode.iloc[0])
            DATASTORE['df'] = df
        return redirect(url_for('missing_data'))

    return render_template_string(missing_tmpl, total_rows=total_rows, missing_cols=list(missing.index), table=table)

@app.route('/outliers')
def outliers():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))

    method = request.args.get('method', 'zscore')
    col = request.args.get('col', '_all_')

    numeric_cols = list(df.select_dtypes(include='number').columns)
    selected_col = None
    mask = pd.Series(False, index=df.index)

    if col and col != '_all_':
        selected_col = col
        if method == 'zscore':
            mask = detect_outliers_zscore(df, col)
        else:
            mask = detect_outliers_iqr(df, col)
    else:
        # apply across all numeric columns (mark row if any numeric column is outlier)
        for c in numeric_cols:
            if method == 'zscore':
                mask = mask | detect_outliers_zscore(df, c)
            else:
                mask = mask | detect_outliers_iqr(df, c)

    out_df = df[mask]
    outlier_count = len(out_df)
    table = out_df.head(200).to_html(classes='min-w-full text-sm', index=False, border=0) if not out_df.empty else '<div class="p-4 text-slate-600">No outliers detected with the selected method/params.</div>'

    return render_template_string(outliers_tmpl, method=method, numeric_cols=numeric_cols, selected_col=selected_col, outlier_count=outlier_count, table=table)

@app.route('/visualize')
def visualize():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))

    cols = list(df.columns)
    vtype = request.args.get('vtype', 'uni')
    xcol = request.args.get('xcol', cols[0] if cols else '')
    ycol = request.args.get('ycol', '')
    kind = request.args.get('kind', 'hist')

    plot_html = None
    try:
        if vtype == 'uni':
            if kind == 'hist':
                fig = px.histogram(df, x=xcol, title=f'Histogram of {xcol}')
            elif kind == 'box':
                fig = px.box(df, y=xcol, title=f'Box of {xcol}')
            elif kind == 'bar':
                fig = px.bar(df[xcol].value_counts().reset_index().rename(columns={'index':xcol, xcol:'count'}), x=xcol, y='count', title=f'Bar of {xcol}')
            else:
                fig = px.histogram(df, x=xcol, title=f'Histogram of {xcol}')

        elif vtype == 'bi' and ycol:
            if kind == 'scatter':
                fig = px.scatter(df, x=xcol, y=ycol, title=f'Scatter: {xcol} vs {ycol}')
            elif kind == 'box':
                fig = px.box(df, x=xcol, y=ycol, title=f'Box {ycol} by {xcol}')
            elif kind == 'bar':
                fig = px.bar(df, x=xcol, y=ycol, title=f'Bar: {xcol} vs {ycol}')
            else:
                fig = px.scatter(df, x=xcol, y=ycol, title=f'Scatter: {xcol} vs {ycol}')
        else:
            # multivariate: show scatter matrix
            fig = px.scatter_matrix(df.select_dtypes(include='number').dropna().iloc[:, :6], title='Scatter matrix (first 6 numeric columns)')

        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        plot_html = f"<div class='p-4 text-red-600'>Failed to create plot: {e}</div>"

    return render_template_string(visualize_tmpl, cols=cols, vtype=vtype, xcol=xcol, ycol=ycol, kind=kind, plot_html=plot_html)

@app.route('/export')
def export_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))
    buf = io.BytesIO()
    # default: csv
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=(DATASTORE.get('filename') or 'data') + '.csv', mimetype='text/csv')

@app.route('/generate_report')
def generate_report():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    df = DATASTORE['df']
    if df is None:
        return redirect(url_for('upload'))

    # Basic summary
    summary = df.describe(include='all').transpose()
    summary_html = summary.to_html(classes='min-w-full text-sm', border=0)

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing>0].to_frame('missing_count')
    missing_html = missing.to_html(classes='min-w-full text-sm', border=0)

    # Outliers brief (zscore across numeric cols)
    numeric = df.select_dtypes(include='number')
    outlier_counts = {}
    for c in numeric.columns:
        mask = detect_outliers_zscore(df, c)
        outlier_counts[c] = int(mask.sum())
    outliers_df = pd.Series(outlier_counts).sort_values(ascending=False).to_frame('zscore_outliers')
    outliers_html = outliers_df.to_html(classes='min-w-full text-sm', border=0)

    # sample plot
    plot_html = ''
    try:
        if len(numeric.columns) >= 1:
            fig = px.histogram(df, x=numeric.columns[0], title=f'Sample: {numeric.columns[0]}')
            plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except Exception:
        plot_html = '<div class="text-slate-600">(plot unavailable)</div>'

    report_html = render_template_string(report_tmpl, filename=DATASTORE.get('filename'), ts=pd.Timestamp.now(), summary_html=summary_html, missing_html=missing_html, outliers_html=outliers_html, plot_html=plot_html)

    # return as downloadable HTML file
    buf = io.BytesIO()
    buf.write(report_html.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=(DATASTORE.get('filename') or 'report') + '_report.html', mimetype='text/html')

# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
