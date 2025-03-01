import streamlit as st
import pandas as pd
import os
from io import BytesIO
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sweetviz as sv
import time
from datetime import datetime
import scipy.stats as stats
from typing import Optional, Dict, List, Tuple

def show_error(message: str) -> None:
    """Display error message with styling"""
    st.markdown(f"""
        <div class='error-message'>
            ❌ {message}
        </div>
    """, unsafe_allow_html=True)

def show_success(message: str) -> None:
    """Display success message with styling"""
    st.markdown(f"""
        <div class='success-message'>
            ✨ {message}
        </div>
    """, unsafe_allow_html=True)

def show_loading() -> None:
    """Display loading spinner"""
    st.markdown("""
        <div style='text-align: center;'>
            <div class='loading-spinner'></div>
        </div>
    """, unsafe_allow_html=True)

def show_loading_animation():
    """Show a loading animation while processing"""
    with st.spinner('Processing...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

def add_tooltip(text: str, help_text: str) -> str:
    """Add tooltip to text"""
    return f"""
        <span title="{help_text}" style="cursor: help; border-bottom: 1px dotted #666;">
            {text}
        </span>
    """

# Configure the Streamlit app's appearance and layout
# 'page_title' sets the browser tab title
# 'layout="wide"' allows more horizontal space, improving the display for tables and graphs
st.set_page_config(
    page_title="DataViz Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/dataviz-pro',
        'Report a bug': "https://github.com/yourusername/dataviz-pro/issues",
        'About': "# DataViz Pro\nA modern data visualization and analysis tool."
    }
)

# Enhanced CSS with modern styling
st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #EC4899;
        --background-color: #F8FAFC;
        --surface-color: #FFFFFF;
        --text-color: #1E293B;
        --accent-color: #10B981;
    }

    /* Card styling */
    .stCard {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    .stCard:hover {
        transform: translateY(-2px);
    }

    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--surface-color);
        padding: 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        flex: 1;
        min-width: 200px;
        border: 1px solid #E2E8F0;
    }
    .metric-card p {
        color: #64748B;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: var(--background-color);
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0.5rem;
        color: var(--text-color);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(124, 58, 237, 0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }

    /* Button styling */
        .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
            border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        }
        .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }

    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 0.5rem;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary-color);
    }

    /* File uploader styling */
    .stFileUploader {
        background: var(--surface-color);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #E2E8F0;
        text-align: center;
    }
    .stFileUploader:hover {
        border-color: var(--primary-color);
    }

    /* Animation classes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Enhanced metric cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--surface-color);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card p {
        color: #64748B;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-trend {
        color: #64748B;
        font-size: 0.75rem;
    }
    
    /* Enhanced cards */
    .stCard {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .stCard:hover {
        transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)

# Updated header with animation
st.markdown("""
    <div class='main-header'>
        <h1>🔍 Advanced Data Sweeper</h1>
        <p>Transform, analyze, and visualize your data with powerful insights</p>
    </div>
""", unsafe_allow_html=True)

# Add after CSS
st.markdown("""
    <div class="fade-in" style="text-align: center; padding: 2rem 0;">
        <h1 style="color: var(--primary-color); font-size: 2.5rem; margin-bottom: 1rem;">
            📊 DataViz Pro
        </h1>
        <p style="color: #64748B; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Transform your data into meaningful insights with our modern visualization and analysis tool.
        </p>
    </div>
""", unsafe_allow_html=True)

# Constants
MAX_FILE_SIZE_MB = 200
ALLOWED_EXTENSIONS = [".csv", ".xlsx"]

def generate_data_summary(df):
    """Generate basic statistical summary and visualizations"""
    summary = {}
    
    # Basic statistics
    summary['basic_stats'] = df.describe()
    
    # Missing values analysis
    summary['missing_values'] = df.isnull().sum()
    
    # Data types
    summary['dtypes'] = df.dtypes
    
    # Column info
    summary['column_info'] = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    # For numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = {
            'mean': df[numeric_cols].mean(),
            'median': df[numeric_cols].median(),
            'std': df[numeric_cols].std(),
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max()
        }
    
    # For categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_stats'] = {
            col: df[col].value_counts().head(5) for col in categorical_cols
        }
    
    return summary

@st.cache_data
def load_data(file: BytesIO) -> Optional[pd.DataFrame]:
    """Cache data loading to improve performance"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        show_detailed_error(
            "File Loading Error",
            str(e),
            ["Check if the file is corrupted",
             "Ensure the file format matches the extension",
             "Try converting to CSV if using Excel"]
        )
        return None

def clean_data(df, options):
    """Advanced data cleaning function"""
    if options.get("remove_duplicates"):
        df = df.drop_duplicates()
    
    if options.get("fill_numeric_nulls"):
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    if options.get("remove_empty_columns"):
        df = df.dropna(axis=1, how='all')
    
    if options.get("strip_whitespace"):
        string_cols = df.select_dtypes(include=['object']).columns
        df[string_cols] = df[string_cols].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    return df

def validate_file(file):
    """Validate file size and type"""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    file_extension = os.path.splitext(file.name)[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False, "Unsupported file type"
    
    return True, ""

def generate_advanced_visualizations(df, column):
    """Generate advanced visualizations for a given column"""
    figures = {}
    
    try:
        # For numeric columns
        if is_numeric_dtype(df[column]):
            # Sample data if it's too large
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > 1000 else df
            
            # Distribution plot
            figures['distribution'] = px.histogram(
                df_sample, 
                x=column,
                title=f'Distribution of {column}',
                template='plotly_white',
                nbins=30  # Limit number of bins for better performance
            )
            
            # Box plot
            figures['box'] = px.box(
                df_sample, 
                y=column,
                title=f'Box Plot of {column}',
                template='plotly_white'
            )
        
        # For all columns (both numeric and categorical)
        # Get value counts with a limit
        value_counts = df[column].value_counts().head(20)  # Limit to top 20 values
        
        # Bar chart for value counts
        figures['bar'] = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=f'Top 20 Values in {column}',
            template='plotly_white',
            labels={'x': column, 'y': 'Count'}
        )
        
        # Only add pie chart if there are few unique values
        if len(value_counts) <= 10:
            figures['pie'] = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {column}',
                template='plotly_white'
            )
    
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")
    
    return figures

def perform_data_analysis(df):
    """Perform advanced data analysis"""
    analysis = {}
    
    # Basic statistics
    analysis['basic_stats'] = df.describe()
    
    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        analysis['correlation'] = df[numeric_cols].corr()
    
    # Missing values analysis
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    analysis['missing_data'] = missing_data
    
    # Duplicate rows analysis
    analysis['duplicates'] = {
        'count': df.duplicated().sum(),
        'percentage': (df.duplicated().sum() / len(df) * 100).round(2)
    }
    
    return analysis

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers

# Replace existing file upload section
st.markdown("""
    <div class="stCard fade-in">
        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
            📂 Upload Your Data
        </h3>
        <p style="color: #64748B; margin-bottom: 1rem;">
            Drag and drop your files here or click to browse. Supports CSV and Excel files.
        </p>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "",  # Empty label as we're using custom HTML above
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help="Maximum file size: 200MB"
)

if uploaded_files:
    for file in uploaded_files:
        tabs = st.tabs([
            "📊 Data Preview",
            "🧹 Data Cleaning",
            "📈 Basic Visualization",
            "🔍 Advanced Analysis",
            "📉 Interactive Plots",
            "💾 Export"
        ])

        with tabs[0]:
            df = load_data(file)
            
            if df is not None:
                # Modern file information card
                st.markdown("""
                    <div class="stCard fade-in" style="margin-bottom: 2rem;">
                        <div style="display: flex; align-items: start; gap: 2rem;">
                            <div style="flex: 1;">
                                <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                                    📄 File Information
                                </h3>
                                <div style="color: #64748B;">
                                    <p><strong>Name:</strong> {filename}</p>
                                    <p><strong>Size:</strong> {filesize:.2f} MB</p>
                                    <p><strong>Upload Time:</strong> {current_time}</p>
                                </div>
                            </div>
                            <div style="flex: 2;">
                                <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                                    📊 Quick Stats
                                </h3>
                                <div class="metric-container">
                                    <div class="metric-card">
                                        <p>Total Rows</p>
                                        <div class="metric-value">{rows:,}</div>
                                        <div class="metric-trend">
                                            {row_health}
                                        </div>
                                    </div>
                                    <div class="metric-card">
                                        <p>Total Columns</p>
                                        <div class="metric-value">{cols}</div>
                                        <div class="metric-trend">
                                            {col_types}
                                        </div>
                                    </div>
                                    <div class="metric-card">
                                        <p>Memory Usage</p>
                                        <div class="metric-value">{memory:.1f} MB</div>
                                        <div class="metric-trend">
                                            {memory_status}
                                        </div>
                                    </div>
                                    <div class="metric-card">
                                        <p>Missing Values</p>
                                        <div class="metric-value">{missing_pct:.1f}%</div>
                                        <div class="metric-trend">
                                            {missing_status}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                """.format(
                    filename=file.name,
                    filesize=file.size / (1024 * 1024),
                    current_time=datetime.now().strftime("%Y-%m-%d %H:%M"),  # Use current time instead
                    rows=len(df),
                    cols=len(df.columns),
                    memory=df.memory_usage().sum() / (1024 * 1024),
                    missing_pct=(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
                    row_health="✅ Good size" if len(df) < 100000 else "⚠️ Large dataset",
                    col_types=f"({len(df.select_dtypes(include=['number']).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical)",
                    memory_status="✅ Optimized" if df.memory_usage().sum() / (1024 * 1024) < 100 else "⚠️ High usage",
                    missing_status="✅ Low" if (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) < 5 else "⚠️ High"
                ), unsafe_allow_html=True)

                # Data Preview Section
                st.markdown("""
                    <div class="stCard fade-in">
                        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                            👀 Data Preview
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add column selector for preview
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_cols: list[str] = st.multiselect(
                        "Select columns to view",
                        df.columns.tolist(),
                        default=df.columns.tolist()[:5]
                    )
                with col2:
                    n_rows = st.number_input(
                        "Number of rows to preview",
                        min_value=5,
                        max_value=100,
                        value=10
                    )

                if selected_cols:
                    st.dataframe(
                        df[selected_cols].head(n_rows),
                        use_container_width=True,
                        height=400
                    )
                    
                # Column Information
                st.markdown("""
                    <div class="stCard fade-in" style="margin-top: 2rem;">
                        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                            📋 Column Information
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                    'Unique Values': df.nunique(),
                    'Sample Values': [', '.join(map(str, df[col].dropna().sample(min(3, df[col].nunique())).tolist())) for col in df.columns]
                })
                
                st.dataframe(
                    col_info,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Null %": st.column_config.ProgressColumn(
                            "Null %",
                            help="Percentage of null values",
                            format="%f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Sample Values": st.column_config.TextColumn(
                            "Sample Values",
                            help="Random sample of values from each column",
                            max_chars=50
                        )
                    }
                )

            else:
                st.error("Failed to load data. Please check the file format and try again.")

        with tabs[1]:
            st.subheader("🧹 Data Cleaning Options")
            
            # Column type conversion
            with st.expander("📊 Convert Column Types", expanded=True):
                col_to_convert: str = st.selectbox("Select column to convert", df.columns)
                new_type = st.selectbox("Convert to", ["numeric", "text", "datetime"])
                
                if st.button("Convert Column", use_container_width=True):
                    try:
                        if new_type == "numeric":
                            # Remove any currency symbols and commas
                            df[col_to_convert] = df[col_to_convert].replace('[\$,]', '', regex=True)
                            # Convert to float
                            df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce')
                            show_success(f"Converted {col_to_convert} to numeric type")
                        elif new_type == "text":
                            df[col_to_convert] = df[col_to_convert].astype(str)
                            show_success(f"Converted {col_to_convert} to text type")
                        elif new_type == "datetime":
                            df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors='coerce')
                            show_success(f"Converted {col_to_convert} to datetime type")
                    except Exception as e:
                        show_error(f"Error converting column: {str(e)}")
            
            # Other cleaning options
            col1, col2 = st.columns(2)
            with col1:
                cleaning_options = {
                    "remove_duplicates": st.checkbox("🔄 Remove duplicates"),
                    "fill_numeric_nulls": st.checkbox("📊 Fill missing numerics with mean"),
                    "remove_empty_columns": st.checkbox("🗑️ Remove empty columns"),
                    "strip_whitespace": st.checkbox("✂️ Strip whitespace"),
                }
            
            with col2:
                cleaning_options.update({
                    "remove_outliers": st.checkbox("📉 Remove outliers"),
                    "convert_dates": st.checkbox("📅 Convert date columns"),
                    "drop_high_null": st.checkbox("❌ Drop columns with high null %"),
                    "round_numerics": st.checkbox("🔢 Round numeric columns")
                })

            if st.button("🧹 Clean Data", use_container_width=True):
                df = clean_data(df, cleaning_options)
                show_success("✨ Data cleaning completed!")

        with tabs[2]:
            st.subheader("📈 Basic Visualization")
            
            # Create two columns for controls and visualization
            control_col, viz_col = st.columns([1, 3])
            
            with control_col:
                # Column selection
                selected_column: str | None = st.selectbox(
                    "Select column to visualize",
                    df.columns.tolist()
                )
                
                # Add visualization type selector
                viz_type = st.selectbox(
                    "Select visualization type",
                    ["Value Counts", "Distribution", "Box Plot"]
                )
            
            with viz_col:
                # Generate visualizations with enhanced size
                if selected_column:
                    figures = generate_advanced_visualizations(df, selected_column)
                    
                    # Display plots with enhanced layout
                    for plot_name, fig in figures.items():
                        # Update layout for each figure
                        fig.update_layout(
                            height=600,  # Increased height
                            width=900,   # Increased width
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(size=12),
                            title_font_size=20,
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            margin=dict(l=40, r=40, t=60, b=40)  # Increased margins
                        )
                        # Display plot with container width
                        st.plotly_chart(fig, use_container_width=True, key=f"basic_viz_{plot_name}_{selected_column}")

        with tabs[3]:
            st.subheader("🔍 Advanced Analysis")
            
            # Create analysis tabs for better organization
            analysis_tabs = st.tabs([
                "📊 Statistical Summary",
                "📉 Distribution Analysis",
                "❌ Missing Data",
                "🔗 Correlation Analysis",
                "🎯 Outlier Detection"
            ])
            
            # Perform analysis
            analysis = perform_data_analysis(df)
            
            with analysis_tabs[0]:
                st.subheader("Statistical Summary")
                
                # Numeric columns summary
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.write("#### Numeric Columns")
                    stats_df = df[numeric_cols].describe()
                    stats_df.loc['skew'] = df[numeric_cols].skew()
                    stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
                    st.dataframe(stats_df.round(2), use_container_width=True)
                
                # Categorical columns summary
                cat_cols = df.select_dtypes(exclude=['number']).columns
                if len(cat_cols) > 0:
                    st.write("#### Categorical Columns")
                    cat_summary = pd.DataFrame({
                        'Column': cat_cols,
                        'Unique Values': [df[col].nunique() for col in cat_cols],
                        'Most Common': [df[col].mode()[0] if not df[col].empty else None for col in cat_cols],
                        'Most Common Count': [df[col].value_counts().iloc[0] if not df[col].empty else 0 for col in cat_cols]
                    })
                    st.dataframe(cat_summary, use_container_width=True)
            
            with analysis_tabs[1]:
                st.subheader("Distribution Analysis")
                
                # Select column for distribution analysis
                dist_col: str = st.selectbox("Select column for distribution analysis", df.columns)
                
                if is_numeric_dtype(df[dist_col]):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution plot
                        fig_dist = px.histogram(
                            df, x=dist_col,
                            title=f"Distribution of {dist_col}",
                            template="plotly_white",
                            marginal="box"  # Add box plot on the margin
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        # QQ plot
                        fig_qq = px.scatter(
                            x=np.sort(df[dist_col]),
                            y=stats.norm.ppf(np.linspace(0.01, 0.99, len(df))),
                            title=f"Q-Q Plot of {dist_col}",
                            template="plotly_white",
                            labels={'x': 'Observed Values', 'y': 'Theoretical Quantiles'}
                        )
                        fig_qq.add_trace(
                            go.Scatter(
                                x=[df[dist_col].min(), df[dist_col].max()],
                                y=[df[dist_col].min(), df[dist_col].max()],
                                mode='lines',
                                line=dict(dash='dash', color='red'),
                                name='Normal Line'
                            )
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                        
                        # Distribution statistics
                        st.write("#### Distribution Statistics")
                        stats_dict = {
                            'Mean': df[dist_col].mean(),
                            'Median': df[dist_col].median(),
                            'Std Dev': df[dist_col].std(),
                            'Skewness': df[dist_col].skew(),
                            'Kurtosis': df[dist_col].kurtosis()
                        }
                        st.json(stats_dict)
                else:
                    # For categorical columns
                    value_counts = df[dist_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Value Distribution of {dist_col}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with analysis_tabs[2]:
                st.subheader("Missing Data Analysis")
                
                # Missing data summary
                missing_data = analysis['missing_data']
                
                # Plot missing data
                fig_missing = px.bar(
                    missing_data,
                    x='Column',
                    y='Percentage',
                    title="Missing Values by Column",
                    template="plotly_white"
                )
                st.plotly_chart(fig_missing, use_container_width=True)
                
                # Detailed missing data table
                st.dataframe(missing_data, use_container_width=True)
            
            with analysis_tabs[3]:
                st.subheader("Correlation Analysis")
                
                if 'correlation' in analysis:
                    # Correlation matrix heatmap
                    fig_corr = px.imshow(
                        analysis['correlation'],
                        title="Correlation Heatmap",
                        template="plotly_white",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Strong correlations table
                    st.write("#### Strong Correlations (|r| > 0.5)")
                    corr_matrix = analysis['correlation']
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i,j]) > 0.5:
                                strong_corr.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Correlation': corr_matrix.iloc[i,j]
                                })
                    if strong_corr:
                        st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                    else:
                        st.info("No strong correlations found")
                else:
                    st.info("No numeric columns available for correlation analysis")
            
            with analysis_tabs[4]:
                st.subheader("Outlier Detection")
                
                # Select column for outlier analysis
                outlier_col: str = st.selectbox("Select column for outlier analysis", df.select_dtypes(include=['number']).columns)
                
                if outlier_col:
                    # Box plot with points
                    fig_box = px.box(
                        df,
                        y=outlier_col,
                        title=f"Box Plot with Outliers: {outlier_col}",
                        template="plotly_white",
                        points="all"  # Show all points
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Outlier statistics
                    outliers = detect_outliers(df, outlier_col)
                    st.write(f"#### Outlier Statistics for {outlier_col}")
                    stats_dict = {
                        'Total Outliers': len(outliers),
                        'Percentage Outliers': f"{(len(outliers) / len(df) * 100):.2f}%",
                        'Min Outlier': float(outliers.min()) if len(outliers) > 0 else None,
                        'Max Outlier': float(outliers.max()) if len(outliers) > 0 else None
                    }
                    st.json(stats_dict)

        with tabs[4]:
            st.subheader("📉 Interactive Plots")
            
            # Create two columns for controls and plot
            control_col, plot_col = st.columns([1, 3])
            
            with control_col:
                plot_type = st.selectbox(
                    "Select plot type",
                    ["Scatter", "Line", "Bar", "Box", "Violin", "3D Scatter", "Histogram"]
                )
                
                if plot_type != "3D Scatter":
                    x_col: str | None = st.selectbox("Select X axis", df.columns)
                    y_col: str | None = st.selectbox("Select Y axis", df.columns)
                    
                    # Additional controls based on plot type
                    if plot_type == "Scatter":
                        color_col: str = st.selectbox("Color by", ["None"] + df.columns.tolist())
                        opacity = st.slider("Point opacity", 0.1, 1.0, 0.7)
                    
                    elif plot_type == "Line":
                        # Corrected line shape options
                        line_shape = st.selectbox(
                            "Line shape", 
                            ["linear", "hv", "vh", "hvh", "vhv"]  # Valid plotly line shapes
                        )
                        line_width = st.slider("Line width", 1, 5, 2)
                    
                    elif plot_type == "Histogram":
                        nbins = st.slider("Number of bins", 5, 100, 30)
                        show_stats = st.checkbox("Show mean/median lines", True)
                else:
                    cols: list[str] = st.multiselect("Select 3 columns", df.columns, max_selections=3)
            
            with plot_col:
                try:
                    fig = None
                    if plot_type != "3D Scatter" and x_col and y_col:
                        if plot_type == "Scatter":
                            fig = px.scatter(
                                df, x=x_col, y=y_col,
                                color=None if color_col == "None" else color_col,
                                title=f"Scatter Plot: {x_col} vs {y_col}",
                                template="plotly_white",
                                opacity=opacity
                            )
                            
                        elif plot_type == "Line":
                            # Create line plot with corrected shape
                            fig = px.line(
                                df, 
                                x=x_col, 
                                y=y_col,
                                title=f"Line Plot: {x_col} vs {y_col}",
                                template="plotly_white"
                            )
                            # Update line properties
                            fig.update_traces(
                                line=dict(
                                    shape=line_shape,  # Apply selected shape
                                    width=line_width   # Apply selected width
                                )
                            )
                        
                        elif plot_type == "Bar":
                            try:
                                # Check if we're dealing with categorical data
                                if not is_numeric_dtype(df[x_col]) or not is_numeric_dtype(df[y_col]):
                                    # For categorical data, show value counts with limit
                                    value_counts = df[x_col].value_counts().head(20)
                                    fig = px.bar(
                                        x=value_counts.index,
                                        y=value_counts.values,
                                        title=f"Top 20 Values in {x_col}",
                                        template="plotly_white",
                                        labels={'x': x_col, 'y': 'Count'}
                                    )
                                else:
                                    # For numeric data, sample if dataset is large
                                    if len(df) > 1000:
                                        df_sample = df.sample(n=1000, random_state=42)
                                    else:
                                        df_sample = df
                                        
                                    fig = px.bar(
                                        df_sample,
                                        x=x_col,
                                        y=y_col,
                                        title=f"Bar Plot: {x_col} vs {y_col}",
                                        template="plotly_white"
                                    )
                                
                                # Enhance bar plot styling
                                fig.update_traces(
                                    marker_line_color='white',
                                    marker_line_width=1,
                                    opacity=0.8
                                )
                                
                            except Exception as e:
                                st.error(f"Error creating bar plot: {str(e)}")
                                continue
                            
                        elif plot_type == "Box":
                            fig = px.box(
                                df, x=x_col, y=y_col,
                                title=f"Box Plot: {x_col} vs {y_col}",
                                template="plotly_white"
                            )
                            
                        elif plot_type == "Violin":
                            fig = px.violin(
                                df, x=x_col, y=y_col,
                                title=f"Violin Plot: {x_col} vs {y_col}",
                                template="plotly_white",
                                box=True
                            )
                            
                        elif plot_type == "Histogram":
                            fig = px.histogram(
                                df, x=x_col,
                                nbins=nbins,
                                title=f"Histogram of {x_col}",
                                template="plotly_white"
                            )
                            if show_stats:
                                mean_val = df[x_col].mean()
                                median_val = df[x_col].median()
                                fig.add_vline(x=mean_val, line_color="red", line_dash="dash", annotation_text="Mean")
                                fig.add_vline(x=median_val, line_color="green", line_dash="dash", annotation_text="Median")
                    
                    elif plot_type == "3D Scatter" and len(cols) == 3:
                        fig = px.scatter_3d(
                            df, x=cols[0], y=cols[1], z=cols[2],
                            title=f"3D Scatter Plot: {', '.join(cols)}",
                            template="plotly_white"
                        )
                    
                    if fig:
                        # Enhanced layout
                        fig.update_layout(
                            height=600,
                            width=900,
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(size=12),
                            title_font_size=20,
                            showlegend=True,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select appropriate columns for the chosen plot type")
                        
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
                    st.info("Try selecting different columns or plot type")

        with tabs[5]:
            st.subheader("💾 Export Options")
            
            # Export format selection
            export_format = st.radio(
                "Select export format:",
                ["CSV", "Excel", "JSON", "Pickle"],
                horizontal=True
            )
            
            # Additional export options
            col1, col2 = st.columns(2)
            with col1:
                include_index = st.checkbox("Include index")
                compress_output = st.checkbox("Compress output")
            
            with col2:
                if export_format == "Excel":
                    sheet_name = st.text_input("Sheet name", "Sheet1")
                elif export_format == "CSV":
                    separator = st.selectbox("Separator", [",", ";", "|", "\t"])
            
            if st.button("⬇️ Export Data", use_container_width=True):
                try:
                    buffer = BytesIO()
                    
                    if export_format == "CSV":
                        df.to_csv(buffer, index=include_index, sep=separator)
                        file_ext = ".csv"
                        mime_type = "text/csv"
                    elif export_format == "Excel":
                        df.to_excel(buffer, index=include_index, sheet_name=sheet_name)
                        file_ext = ".xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    elif export_format == "JSON":
                        df.to_json(buffer)
                        file_ext = ".json"
                        mime_type = "application/json"
                    else:  # Pickle
                        df.to_pickle(buffer)
                        file_ext = ".pkl"
                        mime_type = "application/octet-stream"
                    
                    buffer.seek(0)
                    st.download_button(
                        label=f"⬇️ Download as {export_format}",
                        data=buffer,
                        file_name=f"processed_data_{int(time.time())}{file_ext}",
                        mime=mime_type
                    )
                except Exception as e:
                    show_error(f"Error during export: {str(e)}")

# Footer with animation
st.markdown(f"""
    <div style='text-align: center; padding: 3rem; color: #64748B; animation: fadeIn 0.8s ease-out;'>
        <p>Made with ❤️ by Syed Muhammad Raza Zaidi</p>
        <p style='font-size: 0.8rem;'>Last updated: {datetime.now().strftime("%B %d, %Y")}</p>
    </div>
""", unsafe_allow_html=True)

# Add TypeScript-validated keyboard shortcuts
st.markdown("""
    <script>
        document.addEventListener('keydown', (event: KeyboardEvent) => {
            if ((event.ctrlKey || event.metaKey) && event.key === 's') {
                event.preventDefault();
                const buttons: NodeListOf<HTMLButtonElement> = document.querySelectorAll('button');
                buttons.forEach((button: HTMLButtonElement) => {
                    if (button.textContent?.includes('Export Data')) {
                        button.click();
                    }
                });
            }
        });
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'light',
        'default_plot_height': 600,
        'default_plot_width': 900,
        'max_rows_preview': 1000,
        'auto_clean_data': False
    }

def show_detailed_error(error_type: str, message: str, suggestions: List[str]) -> None:
    """Show detailed error with suggestions"""
    st.error(f"⚠️ {error_type}: {message}")
    if suggestions:
        st.info("💡 Suggestions:")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate dataset and return issues"""
    issues = []
    
    if df.empty:
        issues.append("Dataset is empty")
    
    if df.columns.duplicated().any():
        issues.append("Dataset contains duplicate column names")
    
    if df.isnull().all().any():
        issues.append("Dataset contains columns with all null values")
        
    return len(issues) == 0, issues

def clean_data(df: pd.DataFrame, options: Dict[str, bool]) -> pd.DataFrame:
    """Clean data with progress tracking"""
    progress_bar = st.progress(0)
    total_steps = len([opt for opt in options.values() if opt])
    current_step = 0
    
    try:
        for option, enabled in options.items():
            if enabled:
                if option == "remove_duplicates":
                    df = df.drop_duplicates()
                elif option == "fill_numeric_nulls":
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif option == "remove_empty_columns":
                    df = df.dropna(axis=1, how='all')
                elif option == "strip_whitespace":
                    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
        return df
    except Exception as e:
        show_detailed_error(
            "Data Cleaning Error",
            str(e),
            ["Check column data types",
             "Verify there's no corrupted data",
             "Try cleaning steps individually"]
        )
        return df

def add_advanced_plot_options(fig: go.Figure) -> go.Figure:
    """Add advanced plot customization options"""
    with st.expander("Advanced Plot Options"):
        col1, col2 = st.columns(2)
        with col1:
            title_size = st.slider("Title Size", 10, 30, 20)
            font_family = st.selectbox("Font Family", ["Arial", "Times New Roman", "Helvetica"])
        with col2:
            theme = st.selectbox("Color Theme", ["light", "dark", "custom"])
            grid = st.checkbox("Show Grid", True)
            
        fig.update_layout(
            title_font_size=title_size,
            font_family=font_family,
            template=f"plotly_{theme}" if theme != "custom" else "none",
            showgrid=grid
        )
        return fig

# Add help section in sidebar
with st.sidebar:
    with st.expander("📚 Help & Documentation"):
        st.markdown("""
            ### Quick Start Guide
            1. Upload your data file (CSV or Excel)
            2. Preview your data in the Data Preview tab
            3. Clean your data using the Data Cleaning options
            4. Explore visualizations in the Basic and Interactive tabs
            
            ### Tips
            - Use the 'Convert Column Types' option to fix data type issues
            - Enable 'Auto-clean' for basic data cleaning
            - Export your cleaned data in various formats
        """)

# Add keyboard shortcuts
st.markdown("""
    <script>
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 's') {
                // Save/Export data
                document.querySelector('button:contains("Export Data")').click();
            }
        });
    </script>
""", unsafe_allow_html=True)