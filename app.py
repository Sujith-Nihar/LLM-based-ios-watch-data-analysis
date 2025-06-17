import os, pandas as pd, streamlit as st, plotly.express as px
import google.generativeai as genai

# ─── CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="Smart Health Dashboard", layout="wide")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))  # <-- set your key

# ─── LOAD DATA ───────────────────────────────────────────
RAW = pd.read_csv("apple_health_data.csv",
                  parse_dates=["startDate", "endDate"],
                  low_memory=False)
RAW["value"] = pd.to_numeric(RAW["value"], errors="coerce")
RAW["Date"]  = RAW["startDate"].dt.date

ALIASES = {
    "HKQuantityTypeIdentifierStepCount":   "Steps",
    "HKQuantityTypeIdentifierHeartRate":   "Heart Rate",
    "HKCategoryTypeIdentifierSleepAnalysis": "Sleep",
}
RAW["Metric"] = RAW["type"].map(ALIASES).fillna(RAW["type"])

# ─── SIDEBAR FILTERS ────────────────────────────────────
st.sidebar.header("📂 Filters")

min_d, max_d = RAW["Date"].min(), RAW["Date"].max()
date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)

agg_choice = st.sidebar.selectbox(
    "Aggregation",
    options=["Daily", "Weekly", "Monthly"],
    index=0,
)

# ─── FILTER + AGGREGATE ─────────────────────────────────
mask = (RAW["Date"] >= date_from) & (RAW["Date"] <= date_to)
df = RAW.loc[mask].copy()

def agg_df(metric, how="sum"):
    tmp = df[df["Metric"] == metric][["Date", "value"]].copy()
    tmp.set_index(pd.to_datetime(tmp["Date"]), inplace=True)
    rule = {"Daily":"D", "Weekly":"W", "Monthly":"M"}[agg_choice]
    if how == "sum":
        tmp = tmp["value"].resample(rule).sum()
    else:
        tmp = tmp["value"].resample(rule).mean()
    return tmp.reset_index(name=metric)

steps  = df[(df["Metric"] == "Steps") & (df["unit"] == "count")][["Date", "value"]].copy()
steps.set_index(pd.to_datetime(steps["Date"]), inplace=True)
rule = {"Daily":"D", "Weekly":"W", "Monthly":"M"}[agg_choice]
steps = steps["value"].resample(rule).sum().reset_index(name="Steps")
hr     = df[
    (df["Metric"] == "Heart Rate") &
    (df["unit"].str.contains("count/min", na=False)) &
    (df["value"] > 40) & (df["value"] < 200)
][["Date", "value"]].copy()
hr.set_index(pd.to_datetime(hr["Date"]), inplace=True)
hr = hr["value"].resample(rule).mean().reset_index(name="Heart Rate")
if "Sleep" in df["Metric"].values:
    sleep_dur = df[df["Metric"]=="Sleep"].assign(
        hours=lambda d:(d["endDate"]-d["startDate"]).dt.total_seconds()/3600)
    sleep = sleep_dur[["Date","hours"]]
    sleep.set_index(pd.to_datetime(sleep["Date"]), inplace=True)
    sleep = sleep["hours"].resample({"Daily":"D","Weekly":"W","Monthly":"M"}[agg_choice]).sum()
    sleep = sleep.rename_axis('Date').reset_index(name="hours")
else:
    sleep = pd.DataFrame(columns=["Date","hours"])

# Merge for Gemini
summary = (
    steps.merge(hr, on="Date", how="outer")
         .merge(sleep, on="Date", how="outer")
         .rename(columns={"hours":"Sleep Hours"})
         .fillna(0)
)

# ─── KPI HEADER ─────────────────────────────────────────
st.title("📊 Apple Health Dashboard  |  🤖 Smart Assistant")
k1,k2,k3 = st.columns(3)
k1.metric("Avg Steps", f"{summary['Steps'].mean():,.0f}")
valid_hr = summary[summary["Heart Rate"] > 0]
k2.metric("Avg Heart Rate", f"{valid_hr['Heart Rate'].mean():.1f} bpm")
valid_sleep = summary[summary["Sleep Hours"] > 0]
if not valid_sleep.empty:
    k3.metric("Avg Sleep", f"{valid_sleep['Sleep Hours'].mean():.2f} h")

# ─── PLOTS ──────────────────────────────────────────────
tab1,tab2,tab3 = st.tabs(["🛎 Steps","❤️ Heart Rate","🛌 Sleep"])
with tab1:
    st.plotly_chart(px.line(steps, x="Date", y="Steps", title=f"{agg_choice} Steps"),
                    use_container_width=True)
with tab2:
    st.plotly_chart(px.line(hr, x="Date", y="Heart Rate", title=f"{agg_choice} Avg Heart Rate"),
                    use_container_width=True)
with tab3:
    if not sleep.empty:
        st.plotly_chart(px.bar(sleep, x="Date", y="hours",
                               title=f"{agg_choice} Sleep Hours"),
                        use_container_width=True)
    else:
        st.info("No sleep data in this range.")

# ─── GEMINI CHAT ────────────────────────────────────────
st.divider()
st.subheader("🤖 Ask Gemini About Your Health")

user_q = st.text_area("Question", "Summarize my progress this quarter")
if st.button("Ask"):
    if not os.getenv("GOOGLE_API_KEY", "AIzaSyBCoI9VO1SwMAAI5_1_98iDTFeJ2aamyNg"):
        st.error("Missing GOOGLE_API_KEY environment variable.")
    else:
        with st.spinner("Gemini is thinking..."):
            model = genai.GenerativeModel("gemini-1.5-flash")
            chat  = model.start_chat()

            csv_snippet = summary.to_csv(index=False)
            gemini_prompt = (
                "You are a personal health coach.\n\n"
                f"The user selected data aggregated {agg_choice.lower()} from {date_from} to {date_to}.\n"
                "Here is the CSV table of that period:\n\n"
                f"{csv_snippet}\n\n"
                f"User question: {user_q}\n\n"
                "Provide concise insights, notice trends, and offer one actionable suggestion."
            )

            reply = chat.send_message(gemini_prompt)
            st.success(reply.text)
