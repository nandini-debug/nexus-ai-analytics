import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.ensemble import IsolationForest

# -------------------------------------------------
# LOAD ENV
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Nexus AI Analytics",
    page_icon="🤖",
    layout="wide"
)


# PREMIUM LIGHT UI

st.markdown("""
<style>


.main {

    background: linear-gradient(
        180deg,
        #f8fafc 0%,
        #eef2ff 100%
    );

    color: #111827;
}


html, body, [class*="css"] {

    font-family: 'Inter', sans-serif;
}


.hero-container {

    padding: 35px;

    border-radius: 28px;

    background: linear-gradient(
        135deg,
        rgba(219,234,254,0.95),
        rgba(237,233,254,0.92),
        rgba(245,243,255,0.9)
    );

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.4);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.06),
        0 4px 12px rgba(99,102,241,0.08);

    margin-bottom: 30px;

    transition: all 0.3s ease-in-out;
}

 HERO HOVER 
.hero-container:hover {

    transform: translateY(-2px);

    box-shadow:
        0 16px 40px rgba(0,0,0,0.08),
        0 8px 18px rgba(99,102,241,0.12);
}


.chat-user {

    background: rgba(219,234,254,0.85);

    backdrop-filter: blur(10px);

    padding: 16px;

    border-radius: 18px;

    margin: 12px 0;

    border-left: 6px solid #2563eb;

    color: #111827;

    box-shadow:
        0 4px 12px rgba(37,99,235,0.08);
}

.chat-agent {

    background: rgba(243,232,255,0.92);

    backdrop-filter: blur(10px);

    padding: 16px;

    border-radius: 18px;

    margin: 12px 0;

    border-left: 6px solid #9333ea;

    color: #111827;

    box-shadow:
        0 4px 12px rgba(147,51,234,0.08);
}


div[data-testid="stMetric"] {

    background: rgba(255,255,255,0.85);

    backdrop-filter: blur(14px);

    padding: 22px;

    border-radius: 22px;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 4px 16px rgba(0,0,0,0.05);

    transition: 0.3s ease-in-out;
}

 KPI HOVER 
div[data-testid="stMetric"]:hover {

    transform: translateY(-4px);

    box-shadow:
        0 12px 26px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #ffffff,
        #f8fafc
    );

    border-right: 1px solid #e5e7eb;
}

 SIDEBAR TEXT 
section[data-testid="stSidebar"] * {

    color: #111827 !important;
}


.stButton > button {

    width: 100%;

    background: linear-gradient(
        135deg,
        #dbeafe,
        #ede9fe
    ) !important;

    color: #111827 !important;

    border: none !important;

    border-radius: 14px !important;

    padding: 12px 16px !important;

    font-weight: 600 !important;

    transition: all 0.3s ease-in-out !important;

    box-shadow:
        0 4px 10px rgba(0,0,0,0.05);
}

/ BUTTON HOVER /
.stButton > button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        135deg,
        #bfdbfe,
        #ddd6fe
    ) !important;

    box-shadow:
        0 8px 18px rgba(0,0,0,0.08);
}


button[data-baseweb="tab"] {

    background: rgba(255,255,255,0.78) !important;

    border-radius: 16px !important;

    margin-right: 10px !important;

    padding: 12px 24px !important;

    border: 1px solid #e5e7eb !important;

    color: #374151 !important;

    font-weight: 600 !important;

    transition: all 0.3s ease-in-out !important;

    backdrop-filter: blur(12px);
}

 ACTIVE TAB 
button[data-baseweb="tab"][aria-selected="true"] {

    background: linear-gradient(
        135deg,
        #dbeafe,
        #ede9fe
    ) !important;

    color: #1d4ed8 !important;

    border: 1px solid #c7d2fe !important;

    box-shadow:
        0 6px 14px rgba(59,130,246,0.12);
}

TAB HOVER 
button[data-baseweb="tab"]:hover {

    background: #f8fafc !important;

    transform: translateY(-2px);
}


[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 4px 12px rgba(0,0,0,0.04);
}


.js-plotly-plot {

    border-radius: 20px !important;

    overflow: hidden !important;

    background: white !important;

    padding: 10px !important;

    box-shadow:
        0 4px 14px rgba(0,0,0,0.05);
}


.stAlert {

    border-radius: 16px !important;
}

[data-testid="stChatInput"] {

    border-radius: 18px !important;
}


[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.75);

    padding: 14px;

    border-radius: 18px;

    border: 2px dashed #c7d2fe;
}


::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-track {

    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {

    background: #cbd5e1;

    border-radius: 20px;
}

::-webkit-scrollbar-thumb:hover {

    background: #94a3b8;
}


h1, h2, h3 {

    color: #111827;

    font-weight: 700;
}


.block-container {

    padding-top: 2rem;
    padding-bottom: 2rem;
}


.stSuccess {

    border-radius: 14px !important;
}


.stInfo {

    border-radius: 14px !important;
}

</style>
""", unsafe_allow_html=True)


# SESSION STATE

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = None

if "sample_prompt" not in st.session_state:
    st.session_state["sample_prompt"] = None

if "agent" not in st.session_state:
    st.session_state["agent"] = None


# BUILD AI MODEL

def build_agent():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(" GROQ_API_KEY missing in .env")
        st.stop()

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )


# INIT MODEL

if st.session_state["agent"] is None:
    st.session_state["agent"] = build_agent()


# ANOMALY DETECTION FUNCTION

def detect_anomalies(df):

    num_df = df.select_dtypes(include=np.number)

    num_df = num_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if num_df.shape[0] < 10:
        return None, None

    if num_df.shape[1] < 2:
        return None, None

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(num_df)

    out = df.loc[num_df.index].copy()

    out["anomaly"] = preds

    anomalies = out[
        out["anomaly"] == -1
    ]

    return out, anomalies


# SIDEBAR

with st.sidebar:

    st.title("  Control Panel")

    uploaded_files = st.file_uploader(
        " Upload CSV Files",
        type=["csv"],
        accept_multiple_files=True
    )

    
    # LOAD CSV FILES
    
    if uploaded_files:

        dfs = []

        for file in uploaded_files:

            df_temp = pd.read_csv(file)

            df_temp["source_file"] = file.name

            dfs.append(df_temp)

        combined_df = pd.concat(
            dfs,
            ignore_index=True,
            sort=False
        )

        st.session_state["df"] = combined_df

        st.success(
            f" Loaded {len(uploaded_files)} files"
        )

        st.metric(
            "Rows",
            combined_df.shape[0]
        )

        st.metric(
            "Columns",
            combined_df.shape[1]
        )

        st.dataframe(
            combined_df.head(),
            use_container_width=True
        )

    st.divider()

   
    # POWER BI STYLE FILTERS
    
    if st.session_state["df"] is not None:

        st.subheader(" Power BI Filters")

        df = st.session_state["df"]

        filtered_df = df.copy()

        # NUMERIC FILTERS
        num_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in num_cols[:3]:

            min_val = float(df[col].min())
            max_val = float(df[col].max())

            selected = st.slider(
                f"{col} Range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )

            filtered_df = filtered_df[
                (filtered_df[col] >= selected[0]) &
                (filtered_df[col] <= selected[1])
            ]

        # CATEGORICAL FILTERS
        cat_cols = df.select_dtypes(
            include="object"
        ).columns

        for col in cat_cols[:2]:

            options = list(
                df[col].dropna().unique()
            )

            selected = st.multiselect(
                f"{col} Filter",
                options,
                default=options
            )

            filtered_df = filtered_df[
                filtered_df[col].isin(selected)
            ]

        st.session_state["filtered_df"] = filtered_df

        st.success(" Filters Applied")

        st.metric(
            "Filtered Rows",
            filtered_df.shape[0]
        )

    st.divider()

    
    # SAMPLE QUESTIONS
    
    st.subheader(" Sample Questions")

    questions = [ "Give dataset overview", 
                 "Show missing values and data quality issues", 
                 "Show strongest correlations", 
                 "Find anomalies and unusual patterns", 
                 "Identify key business trends", 
                 "Provide executive-level business insights", 
                 "Generate strategic recommendations" ]

    for q in questions:

        if st.button(
            q,
            use_container_width=True
        ):

            st.session_state["sample_prompt"] = q

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
st.title("🤖 Nexus AI Analytics Platform")



# -------------------------------------------------
# GET FILTERED DATA
# -------------------------------------------------
df = st.session_state.get(
    "filtered_df",
    st.session_state["df"]
)

if df is None:

    st.warning(
        " Upload CSV dataset first"
    )

    st.stop()

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 AI Charts",
    "🚨 Anomalies",
    "🤖 AI Chat"
])


# TAB 1 - DASHBOARD

with tab1:

    st.subheader(" Executive Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.markdown("---")

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(num_cols) > 0:

        col = num_cols[0]

        left, right = st.columns(2)

        left.plotly_chart(
            px.bar(
                df,
                y=col,
                title=" Value Overview"
            ),
            use_container_width=True
        )

        right.plotly_chart(
            px.box(
                df,
                y=col,
                title=" Distribution"
            ),
            use_container_width=True
        )


# TAB 2 - AI CHARTS------
with tab2:

    st.subheader(" AI Generated Charts")

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(num_cols) == 0:

        st.warning(
            " No numeric columns found"
        )

    else:

        # DISTRIBUTION
        st.markdown(
            " Distribution Analysis"
        )

        st.plotly_chart(
            px.histogram(
                df,
                x=num_cols[0],
                title=f"Distribution of {num_cols[0]}"
            ),
            use_container_width=True
        )

        st.info(
            f" Average {num_cols[0]} = "
            f"{df[num_cols[0]].mean():,.2f}"
        )

        # RELATIONSHIP
        if len(num_cols) >= 2:

            st.markdown(
                " Relationship Analysis"
            )

            st.plotly_chart(
                px.scatter(
                    df,
                    x=num_cols[0],
                    y=num_cols[1],
                    title="Correlation Pattern"
                ),
                use_container_width=True
            )

            corr_value = df[
                num_cols[0]
            ].corr(
                df[num_cols[1]]
            )

            st.info(
                f"Correlation = {corr_value:.2f}"
            )

        # HEATMAP
        if len(num_cols) > 1:

            st.markdown(
                "### 🔥 Correlation Heatmap"
            )

            corr = df[num_cols].corr()

            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                annotation_text=corr.round(2).values,
                showscale=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                "✅ AI detected relationships between variables"
            )


# TAB 3 - ANOMALIES ONLY

with tab3:

    st.subheader(
        " Anomaly Detection (Isolation Forest)"
    )

    result, anomalies = detect_anomalies(df)

    if result is None or anomalies is None:

        st.warning(
            " Need at least 2 numeric columns and enough rows"
        )

    else:

        c1, c2 = st.columns(2)

        c1.metric(
            "Total Rows",
            len(result)
        )

        c2.metric(
            "Anomalies",
            len(anomalies)
        )

        if len(anomalies) > 0:

            st.error(
                f" Detected {len(anomalies)} anomalies"
            )

            st.dataframe(
                anomalies,
                use_container_width=True
            )

            num_cols = result.select_dtypes(
                include=np.number
            ).columns

            if len(num_cols) >= 2:

                fig = px.scatter(
                    result,
                    x=num_cols[0],
                    y=num_cols[1],
                    color="anomaly",
                    title=" Anomaly Visualization"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.info(
                    " Red points indicate unusual patterns"
                )

        else:

            st.success(
                " No anomalies detected"
            )


# TAB 4 - AI CHAT

with tab4:

    st.subheader(" AI CHAT")

    # SHOW CHAT HISTORY
    for msg in st.session_state["chat_history"]:

        if msg["role"] == "user":

            st.markdown(
                f"""
<div class="chat-user">
 {msg["content"]}
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="chat-agent">
 {msg["content"]}
</div>
""",
                unsafe_allow_html=True
            )

    # CHAT INPUT
    user_input = st.chat_input(
        "Ask anything about your dataset..."
    )

    # SAMPLE QUESTION
    if st.session_state["sample_prompt"]:

        user_input = st.session_state[
            "sample_prompt"
        ]

        st.session_state["sample_prompt"] = None

    # PROCESS INPUT
    if user_input:

        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_input
        })

        with st.spinner(
            "🤖 AI analyzing dataset..."
        ):

            prompt = f"""
You are an expert AI CHAT.

Dataset Shape:
{df.shape}

Columns:
{list(df.columns)}

Missing Values:
{df.isnull().sum().to_string()}

Preview:
{df.head(5).to_string()}

User Question:
{user_input}

Instructions:
- Analyze carefully
- Give business insights
- Explain trends
- Mention risks
- Respond professionally
"""

            try:

                response = st.session_state[
                    "agent"
                ].invoke(prompt).content

            except Exception as e:

                response = f" Error: {str(e)}"

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response
        })

        st.markdown(
            f"""
<div class="chat-agent">
 {response}
</div>
""",
            unsafe_allow_html=True
        )