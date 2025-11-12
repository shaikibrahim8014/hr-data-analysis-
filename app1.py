import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-message {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "data" not in st.session_state:
    st.session_state.data = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "outliers" not in st.session_state:
    st.session_state.outliers = {}
if "preprocessing_results" not in st.session_state:
    st.session_state.preprocessing_results = None


# Authentication function
def authenticate():
    st.markdown(
        '<h1 class="main-header">🔐 Data Analysis Platform</h1>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Sign in to access your dashboard")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            submit_button = st.form_submit_button("Sign In", use_container_width=True)

            if submit_button:
                if username and password:
                    st.session_state.authenticated = True
                    st.success(
                        "✅ Login successful! Welcome to the Data Analysis Platform."
                    )
                    st.rerun()
                else:
                    st.error("❌ Please enter both username and password.")


# Data upload and processing functions
def upload_data():
    st.markdown("## 📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your dataset in CSV format for analysis",
    )

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            if not df.empty:
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                st.session_state.outliers = {}

                st.markdown(
                    '<div class="success-message">✅ File uploaded successfully!</div>',
                    unsafe_allow_html=True,
                )

                # Display basic info
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Total Rows", len(df))
                with col2:
                    st.metric("📋 Total Columns", len(df.columns))
                with col3:
                    st.metric("❌ Missing Values", int(df.isnull().sum().sum()))
                with col4:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.metric("🔢 Numeric Columns", numeric_cols)

                return True
            else:
                st.error("❌ The uploaded file is empty.")
                return False

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return False

    return False


def display_data_overview():
    if st.session_state.data is not None:
        df = st.session_state.processed_data

        st.markdown("## 📊 Dataset Overview")

        # Dataset statistics
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Dataset Information")
            info_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str),
                    "Non-Null Count": df.count().values,
                    "Null Count": df.isnull().sum().values,
                    "Unique Values": df.nunique().values,
                }
            )
            st.dataframe(info_df, use_container_width=True)

        with col2:
            st.markdown("### Quick Statistics")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(
                f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            )
            st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")

            # Data quality score
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_score = (
                ((total_cells - missing_cells) / total_cells) * 100
                if total_cells > 0
                else 0
            )
            st.metric("🎯 Data Quality Score", f"{quality_score:.1f}%")


def view_raw_data():
    if st.session_state.data is not None:
        df = st.session_state.processed_data

        st.markdown("## 👀 Raw Data View")

        col1, col2 = st.columns([3, 1])

        with col2:
            show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
            show_info = st.checkbox("Show data types", value=True)

        with col1:
            st.dataframe(df.head(show_rows), use_container_width=True)

        if show_info:
            st.markdown("### Data Types and Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Numeric Columns:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.write("No numeric columns found.")

            with col2:
                st.markdown("**Categorical Columns:**")
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                if categorical_cols:
                    cat_info = pd.DataFrame(
                        {
                            "Column": categorical_cols,
                            "Unique Values": [
                                df[col].nunique() for col in categorical_cols
                            ],
                            "Most Frequent": [
                                (
                                    df[col].mode().iloc[0]
                                    if not df[col].mode().empty
                                    else "N/A"
                                )
                                for col in categorical_cols
                            ],
                        }
                    )
                    st.dataframe(cat_info, use_container_width=True)
                else:
                    st.write("No categorical columns found.")


def data_preprocessing():
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data

        st.markdown("## 🔧 Data Preprocessing")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔄 Handle Missing Values", use_container_width=True):
                handle_missing_values()

        with col2:
            if st.button("🎯 Normalize Data", use_container_width=True):
                normalize_data()

        with col3:
            if st.button("🏷️ Encode Categorical", use_container_width=True):
                encode_categorical()

        with col4:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.processed_data = st.session_state.data.copy()
                st.session_state.outliers = {}
                st.success("✅ Data reset to original state!")
                st.rerun()

        # Show preprocessing results
        if st.session_state.preprocessing_results:
            st.markdown(
                '<div class="success-message">'
                + st.session_state.preprocessing_results
                + "</div>",
                unsafe_allow_html=True,
            )


def handle_missing_values():
    df = st.session_state.processed_data.copy()
    missing_before = df.isnull().sum().sum()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Handle numeric columns with mean
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Handle categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = (
                df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            )
            df[col].fillna(mode_value, inplace=True)

    missing_after = df.isnull().sum().sum()
    handled = int(missing_before - missing_after)

    st.session_state.processed_data = df
    st.session_state.preprocessing_results = (
        f"✅ Successfully handled {handled} missing values using mean/mode imputation."
    )


def normalize_data():
    df = st.session_state.processed_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))

        st.session_state.processed_data = df
        st.session_state.preprocessing_results = f"✅ Successfully normalized {len(numeric_cols)} numeric columns using StandardScaler."
    else:
        st.session_state.preprocessing_results = (
            "⚠️ No numeric columns found for normalization."
        )


def encode_categorical():
    df = st.session_state.processed_data
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    encoded_count = 0

    for col in categorical_cols:
        if df[col].nunique() < 20:  # Only encode if reasonable number of categories
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_count += 1

    st.session_state.processed_data = df
    st.session_state.preprocessing_results = f"✅ Successfully encoded {encoded_count} categorical columns using Label Encoding."


def outlier_detection():
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        st.markdown("## 🎯 Outlier Detection")

        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns available for outlier detection.")
            return

        col1, col2 = st.columns(2)

        with col1:
            method = st.selectbox(
                "Select Detection Method", ["IQR Method", "Z-Score Method"]
            )
            selected_column = st.selectbox("Select Column", numeric_cols)

        with col2:
            if method == "Z-Score Method":
                threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
            else:
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)

        if st.button("🔍 Detect Outliers", use_container_width=True):
            if method == "IQR Method":
                outliers = detect_outliers_iqr(df, selected_column, multiplier)
            else:
                outliers = detect_outliers_zscore(df, selected_column, threshold)

            st.session_state.outliers[selected_column] = outliers

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### Outliers in {selected_column}")
                if len(outliers) > 0:
                    outlier_df = pd.DataFrame(
                        {
                            "Index": outliers,
                            "Value": df.loc[outliers, selected_column].values,
                        }
                    )
                    st.dataframe(outlier_df, use_container_width=True)
                else:
                    st.success("✅ No outliers detected!")

            with col2:
                st.metric("🎯 Outliers Found", len(outliers))
                st.metric("📊 Percentage", f"{len(outliers)/len(df)*100:.2f}%")

                if len(outliers) > 0:
                    if st.button("🗑️ Remove Outliers"):
                        df_cleaned = df.drop(index=outliers)
                        st.session_state.processed_data = df_cleaned
                        # remove the outliers info for that column
                        st.session_state.outliers.pop(selected_column, None)
                        st.success(f"✅ Removed {len(outliers)} outliers!")
                        st.rerun()


def detect_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = df[
        (df[column] < lower_bound) | (df[column] > upper_bound)
    ].index.tolist()
    return outliers


def detect_outliers_zscore(df, column, threshold=3):
    # compute z-scores on non-null values and map back to original indices
    series = df[column]
    non_null = series.dropna()
    if non_null.empty:
        return []
    z = np.abs(stats.zscore(non_null))
    outlier_mask = z > threshold
    outlier_indices = non_null.index[outlier_mask].tolist()
    return outlier_indices


def data_visualization():
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data

        st.markdown("## 📈 Data Visualization")

        viz_type = st.selectbox(
            "Select Visualization Type",
            [
                "Univariate Analysis",
                "Bivariate Analysis",
                "Multivariate Analysis",
                "Custom Plot",
            ],
        )

        if viz_type == "Univariate Analysis":
            univariate_analysis(df)
        elif viz_type == "Bivariate Analysis":
            bivariate_analysis(df)
        elif viz_type == "Multivariate Analysis":
            multivariate_analysis(df)
        else:
            custom_plots(df)


def univariate_analysis(df):
    st.markdown("### 📊 Univariate Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    col1, col2 = st.columns(2)

    with col1:
        if len(numeric_cols) > 0:
            selected_numeric = st.selectbox("Select Numeric Column", numeric_cols)

            # Histogram
            fig_hist = px.histogram(
                df,
                x=selected_numeric,
                nbins=30,
                title=f"Distribution of {selected_numeric}",
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Box plot
            fig_box = px.box(
                df, y=selected_numeric, title=f"Box Plot of {selected_numeric}"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        if len(categorical_cols) > 0:
            selected_categorical = st.selectbox(
                "Select Categorical Column", categorical_cols
            )

            # Value counts
            value_counts = df[selected_categorical].value_counts().head(10)

            # Bar chart
            fig_bar = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top 10 Values in {selected_categorical}",
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {selected_categorical}",
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)


def bivariate_analysis(df):
    st.markdown("### 📊 Bivariate Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox("Select X-axis", all_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis", all_cols)

    if x_col != y_col:
        # Determine plot type based on data types
        x_is_numeric = x_col in numeric_cols
        y_is_numeric = y_col in numeric_cols

        if x_is_numeric and y_is_numeric:
            # Scatter plot
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")

            # Add correlation coefficient
            try:
                correlation = df[x_col].corr(df[y_col])
                fig.add_annotation(
                    text=f"Correlation: {correlation:.3f}",
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                )
            except Exception:
                pass

        elif x_is_numeric and not y_is_numeric:
            # Box plot
            fig = px.box(df, x=y_col, y=x_col, title=f"{x_col} by {y_col}")

        elif not x_is_numeric and y_is_numeric:
            # Box plot
            fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")

        else:
            # Heatmap for categorical vs categorical
            crosstab = pd.crosstab(df[x_col], df[y_col])
            fig = px.imshow(
                crosstab, title=f"Cross-tabulation: {x_col} vs {y_col}", aspect="auto"
            )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def multivariate_analysis(df):
    st.markdown("### 📊 Multivariate Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for multivariate analysis.")
        return

    # Correlation matrix
    st.markdown("#### Correlation Matrix")
    corr_matrix = df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Pair plot (for small number of columns)
    if len(numeric_cols) <= 5:
        st.markdown("#### Pair Plot")

        # Create scatter matrix
        fig_scatter = px.scatter_matrix(df[numeric_cols])
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Principal Component Analysis (if applicable)
    if len(numeric_cols) >= 3:
        st.markdown("#### Principal Component Analysis")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        X = df[numeric_cols].dropna()
        if X.shape[0] > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            # Explained variance plot
            fig_pca = px.bar(
                x=range(1, len(pca.explained_variance_ratio_) + 1),
                y=pca.explained_variance_ratio_,
                title="PCA - Explained Variance Ratio",
            )
            fig_pca.update_xaxes(title="Principal Component")
            fig_pca.update_yaxes(title="Explained Variance Ratio")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("No complete numeric rows available for PCA.")


def custom_plots(df):
    st.markdown("### 🎨 Custom Plots")

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Line Plot", "Area Plot", "Violin Plot", "3D Scatter", "Sunburst"],
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_cols = df.columns.tolist()

    if plot_type == "Line Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", all_cols, key="line_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")

        if st.button("Generate Line Plot"):
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "3D Scatter" and len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
        with col3:
            z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")

        if st.button("Generate 3D Scatter"):
            fig = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                title=f"3D Scatter: {x_col}, {y_col}, {z_col}",
            )
            st.plotly_chart(fig, use_container_width=True)


def generate_report():
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data

        st.markdown("## 📋 Analysis Report")

        # Generate comprehensive report
        report_data = {
            "Dataset Summary": {
                "Total Rows": len(df),
                "Total Columns": len(df.columns),
                "Missing Values": int(df.isnull().sum().sum()),
                "Duplicate Rows": int(df.duplicated().sum()),
                "Memory Usage (KB)": df.memory_usage(deep=True).sum() / 1024,
            },
            "Column Information": {
                "Numeric Columns": int(
                    len(df.select_dtypes(include=[np.number]).columns)
                ),
                "Categorical Columns": int(
                    len(df.select_dtypes(include=["object", "category"]).columns)
                ),
                "Boolean Columns": int(len(df.select_dtypes(include=["bool"]).columns)),
                "DateTime Columns": int(
                    len(df.select_dtypes(include=["datetime64"]).columns)
                ),
            },
        }

        # Display report sections
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Dataset Summary")
            for key, value in report_data["Dataset Summary"].items():
                if isinstance(value, float):
                    st.metric(key, f"{value:.2f}")
                else:
                    st.metric(key, value)

        with col2:
            st.markdown("### 📋 Column Information")
            for key, value in report_data["Column Information"].items():
                st.metric(key, value)

        # Data quality assessment
        st.markdown("### 🎯 Data Quality Assessment")

        total_cells = len(df) * len(df.columns)
        missing_cells = int(df.isnull().sum().sum())
        completeness = (
            ((total_cells - missing_cells) / total_cells) * 100
            if total_cells > 0
            else 0
        )

        total_outliers = sum(len(v) for v in st.session_state.outliers.values())
        outlier_pct = (total_outliers / len(df) * 100) if len(df) > 0 else 0

        data_quality_score = min(
            100, completeness + max(0, (100 - outlier_pct))
        )  # heuristic

        quality_metrics = {
            "Data Completeness": f"{completeness:.1f}%",
            "Outliers Detected": total_outliers,
            "Data Quality Score": f"{data_quality_score:.1f}%",
        }

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (key, value) in enumerate(quality_metrics.items()):
            with cols[i]:
                st.metric(key, value)

        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = generate_recommendations(df)

        for rec in recommendations:
            st.markdown(f"• {rec}")

        # Export functionality
        st.markdown("### 📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Download Processed Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        with col2:
            if st.button("📋 Download Report"):
                report_text = generate_text_report(
                    df, report_data, quality_metrics, recommendations
                )
                st.download_button(
                    label="Download Report",
                    data=report_text,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

        with col3:
            if st.button("📈 Download Summary Stats"):
                summary_stats = df.describe()
                csv_stats = summary_stats.to_csv()
                st.download_button(
                    label="Download Stats",
                    data=csv_stats,
                    file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


def generate_recommendations(df):
    recommendations = []

    # Missing values
    total_cells = len(df) * len(df.columns) if len(df) > 0 else 0
    missing_pct = (
        (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
    )
    if missing_pct > 5:
        recommendations.append(
            f"High percentage of missing values ({missing_pct:.1f}%). Consider data imputation or collection improvement."
        )

    # Outliers
    total_outliers = sum(
        len(outliers) for outliers in st.session_state.outliers.values()
    )
    if total_outliers > len(df) * 0.05:
        recommendations.append(
            "Significant number of outliers detected. Investigate data collection process."
        )

    # Data size
    if len(df) < 100:
        recommendations.append(
            "Small dataset size. Consider collecting more data for robust analysis."
        )

    # Column types
    numeric_ratio = (
        len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        if len(df.columns) > 0
        else 0
    )
    if numeric_ratio < 0.3:
        recommendations.append(
            "Low proportion of numeric columns. Consider feature engineering or encoding."
        )

    # Duplicates
    if df.duplicated().sum() > 0:
        recommendations.append(
            f"Found {df.duplicated().sum()} duplicate rows. Consider removing duplicates."
        )

    if not recommendations:
        recommendations.append("Dataset appears to be in good condition for analysis!")

    return recommendations


def generate_text_report(df, report_data, quality_metrics, recommendations):
    report = f"""
DATA ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
===============
"""

    for key, value in report_data["Dataset Summary"].items():
        report += f"{key}: {value}\n"

    report += f"""
COLUMN INFORMATION
==================
"""

    for key, value in report_data["Column Information"].items():
        report += f"{key}: {value}\n"

    report += f"""
DATA QUALITY METRICS
====================
"""

    for key, value in quality_metrics.items():
        report += f"{key}: {value}\n"

    report += f"""
RECOMMENDATIONS
===============
"""

    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"

    return report


# Main application
def main():
    if not st.session_state.authenticated:
        authenticate()
    else:
        # Sidebar navigation
        st.sidebar.markdown("## 🧭 Navigation")

        # Logout button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.data = None
            st.session_state.processed_data = None
            st.session_state.outliers = {}
            st.rerun()

        # Main navigation
        page = st.sidebar.selectbox(
            "Select Page",
            [
                "📁 Upload Data",
                "👀 View Data",
                "🔧 Preprocessing",
                "🎯 Outlier Detection",
                "📈 Visualization",
                "📋 Generate Report",
            ],
        )

        # Display current data info in sidebar
        if st.session_state.data is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Current Dataset")
            st.sidebar.write(f"**Rows:** {len(st.session_state.processed_data)}")
            st.sidebar.write(
                f"**Columns:** {len(st.session_state.processed_data.columns)}"
            )
            st.sidebar.write(
                f"**Missing:** {int(st.session_state.processed_data.isnull().sum().sum())}"
            )

        # Main content area
        st.markdown(
            '<h1 class="main-header">📊 Data Analysis Platform</h1>',
            unsafe_allow_html=True,
        )

        # Route to appropriate page
        if page == "📁 Upload Data":
            upload_data()
            if st.session_state.data is not None:
                display_data_overview()

        elif page == "👀 View Data":
            if st.session_state.data is not None:
                view_raw_data()
            else:
                st.warning("⚠️ Please upload data first!")

        elif page == "🔧 Preprocessing":
            if st.session_state.data is not None:
                data_preprocessing()
            else:
                st.warning("⚠️ Please upload data first!")

        elif page == "🎯 Outlier Detection":
            if st.session_state.data is not None:
                outlier_detection()
            else:
                st.warning("⚠️ Please upload data first!")

        elif page == "📈 Visualization":
            if st.session_state.data is not None:
                data_visualization()
            else:
                st.warning("⚠️ Please upload data first!")

        elif page == "📋 Generate Report":
            if st.session_state.data is not None:
                generate_report()
            else:
                st.warning("⚠️ Please upload data first!")


if __name__ == "__main__":
    main()
    
