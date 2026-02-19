import os
import re
import textwrap
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.makedirs("/tmp/mplconfig", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components

_BNP_PRIMARY = "#167a5f"
_BNP_DARK = "#0f5b46"
_BNP_MID = "#2f8f74"
_BNP_SOFT = "#6cb79f"
_BNP_TINT = "#d8ece6"
_BNP_BG = "#f4faf8"
_BNP_PANEL = "#ffffff"
_BNP_GRID = "#d7e8e2"
_COLOR_A = _BNP_PRIMARY
_COLOR_B = _BNP_MID
_TEMPLATE = "bnp_white"
_BANNER_PATH = "BNP-Paribas-bureaux.jpg"

pio.templates[_TEMPLATE] = go.layout.Template(
    layout=go.Layout(
        colorway=[_BNP_PRIMARY, _BNP_MID, _BNP_SOFT, _BNP_DARK, "#8ac8b4", "#b8ddd0"],
        paper_bgcolor=_BNP_PANEL,
        plot_bgcolor=_BNP_PANEL,
        font=dict(color=_BNP_DARK),
        xaxis=dict(gridcolor=_BNP_GRID, zerolinecolor=_BNP_GRID, linecolor=_BNP_GRID),
        yaxis=dict(gridcolor=_BNP_GRID, zerolinecolor=_BNP_GRID, linecolor=_BNP_GRID),
    )
)
px.defaults.template = _TEMPLATE
px.defaults.color_discrete_sequence = [_BNP_PRIMARY, _BNP_MID, _BNP_SOFT, _BNP_DARK]

st.set_page_config(page_title="SR Dashboard", layout="wide")
st.markdown(
    f"""
    <style>
        :root {{
            --bnp-primary: {_BNP_PRIMARY};
            --bnp-dark: {_BNP_DARK};
            --bnp-soft: {_BNP_SOFT};
            --bnp-bg: {_BNP_BG};
            --bnp-grid: {_BNP_GRID};
        }}
        .stApp {{
            background: radial-gradient(circle at 12% 8%, #ffffff 0%, {_BNP_BG} 50%, #e8f4f0 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #f7fcfa 0%, #e8f4f0 100%);
            border-right: 1px solid #cfe4dc;
        }}
        [data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid #d3e7e0;
            border-radius: 12px;
            padding: 8px 12px;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: var(--bnp-dark);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: #e8f4f0;
            color: var(--bnp-dark);
            border-radius: 10px;
            border: 1px solid #d3e7e0;
            padding: 6px 14px;
        }}
        .stTabs [aria-selected="true"] {{
            background: var(--bnp-primary) !important;
            color: #ffffff !important;
            border-color: var(--bnp-primary) !important;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid #d3e7e0;
            border-radius: 12px;
            background: #ffffff;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: var(--bnp-dark);
        }}
        .stImage img {{
            border-radius: 14px;
            border: 1px solid #cfe4dc;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def fmt_pct(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "n/a"


def fmt_days(value: float) -> str:
    return f"{value:.2f} d" if pd.notna(value) else "n/a"


def wrap_label(value: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(str(value), width))


def normalize_data_source(path: str) -> str:
    raw = str(path).strip()
    if not raw:
        return raw

    if "drive.google.com" not in raw:
        return raw

    parsed = urlparse(raw)
    file_id = None

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        file_id = match.group(1)
    else:
        query = parse_qs(parsed.query)
        file_id = query.get("id", [None])[0]

    if not file_id:
        return raw

    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    src = normalize_data_source(path)
    return pd.read_parquet(src)


@st.cache_data(show_spinner=False)
def load_html(path: str) -> str:
    src = normalize_data_source(path)
    if src.startswith(("http://", "https://")):
        resp = requests.get(src, timeout=180)
        resp.raise_for_status()
        return resp.text
    return Path(src).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_sr_overdue_map(path: str) -> pd.DataFrame:
    src = normalize_data_source(path)
    base = pd.read_parquet(src, columns=["ID", "OVERDUE_FLAG_ASOF"])
    base["ID"] = pd.to_numeric(base["ID"], errors="coerce").astype("Int64")
    base["OVERDUE_FLAG_ASOF"] = pd.to_numeric(base["OVERDUE_FLAG_ASOF"], errors="coerce")
    base = base.dropna(subset=["ID"])
    base = (
        base.groupby("ID", as_index=False)["OVERDUE_FLAG_ASOF"]
        .mean()
    )
    return base


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_q90(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(s.quantile(0.90)) if len(s) else np.nan


def prepare_handoff_data(
    activity_df: pd.DataFrame,
    sr_overdue_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str | None]:
    group_col = pick_first_existing_column(activity_df, ["JUR_ASSIGNEDGROUP_ID", "JUR_DESK_ID"])
    time_col = pick_first_existing_column(activity_df, ["CREATIONDATE", "ACCEPTED_DATE", "UPDATE_DATE"])
    if group_col is None or time_col is None or "SR_ID" not in activity_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Missing required columns to compute handoffs."

    work = activity_df.copy()
    for col in ["CREATIONDATE", "ACCEPTED_DATE", "UPDATE_DATE", "CLOSINGDATE"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")

    needed = ["SR_ID", group_col, time_col]
    work = work.dropna(subset=needed).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "No valid rows after cleaning handoff inputs."

    sort_cols = ["SR_ID", time_col]
    if "ID" in work.columns:
        sort_cols.append("ID")
    work = work.sort_values(sort_cols)

    work["PREV_GROUP"] = work.groupby("SR_ID")[group_col].shift(1)
    work["HANDOFF"] = work["PREV_GROUP"].notna() & (work[group_col] != work["PREV_GROUP"])

    task_col = "ID" if "ID" in work.columns else group_col
    sr_handoffs = (
        work.groupby("SR_ID")
        .agg(
            n_tasks=(task_col, "count"),
            n_groups=(group_col, pd.Series.nunique),
            n_handoffs=("HANDOFF", "sum"),
            first_step=(time_col, "min"),
        )
        .reset_index()
    )
    sr_handoffs["SR_ID"] = pd.to_numeric(sr_handoffs["SR_ID"], errors="coerce").astype("Int64")

    if sr_overdue_map is not None and not sr_overdue_map.empty:
        overdue_map = sr_overdue_map.rename(columns={"ID": "SR_ID"})
        sr_handoffs = sr_handoffs.merge(overdue_map, on="SR_ID", how="left")
    else:
        sr_handoffs["OVERDUE_FLAG_ASOF"] = np.nan

    if "CLOSINGDATE" in work.columns:
        close_end = work.groupby("SR_ID")["CLOSINGDATE"].max().rename("last_close")
        sr_handoffs = sr_handoffs.merge(close_end.reset_index(), on="SR_ID", how="left")
        sr_handoffs["is_closed"] = sr_handoffs["last_close"].notna().astype(int)
        sr_handoffs["close_delay_d"] = (
            (sr_handoffs["last_close"] - sr_handoffs["first_step"]).dt.total_seconds() / 86400
        )
        sr_handoffs.loc[sr_handoffs["close_delay_d"] < 0, "close_delay_d"] = np.nan
    else:
        sr_handoffs["is_closed"] = np.nan
        sr_handoffs["close_delay_d"] = np.nan

    bins = [-0.5, 0.5, 1.5, 3.5, 999]
    labels = ["0", "1", "2-3", "4+"]
    sr_handoffs["HANDOFF_BUCKET"] = pd.cut(sr_handoffs["n_handoffs"].fillna(0), bins=bins, labels=labels)

    impact = (
        sr_handoffs.groupby("HANDOFF_BUCKET", observed=False)
        .agg(
            n_sr=("SR_ID", "count"),
            close_med=("close_delay_d", "median"),
            close_p90=("close_delay_d", safe_q90),
            overdue_rate=("OVERDUE_FLAG_ASOF", safe_mean),
            open_rate=("is_closed", lambda s: 1 - safe_mean(s)),
        )
        .reindex(labels)
        .reset_index()
        .rename(columns={"index": "HANDOFF_BUCKET"})
    )
    impact["n_sr"] = pd.to_numeric(impact["n_sr"], errors="coerce").fillna(0).astype(int)

    transitions = (
        work[work["HANDOFF"]]
        .groupby(["PREV_GROUP", group_col])
        .size()
        .reset_index(name="n")
        .rename(columns={"PREV_GROUP": "FROM", group_col: "TO"})
    )
    if not transitions.empty:
        transitions["FROM"] = pd.to_numeric(transitions["FROM"], errors="coerce").astype("Int64")
        transitions["TO"] = pd.to_numeric(transitions["TO"], errors="coerce").astype("Int64")

    meta = {
        "group_col": group_col,
        "time_col": time_col,
        "n_rows": int(len(work)),
        "n_sr": int(sr_handoffs["SR_ID"].nunique()),
    }
    return sr_handoffs, impact, transitions, meta, None


@st.cache_data(show_spinner=False)
def history_summary(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    field_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    weekly_field_counts: dict[tuple[pd.Timestamp, str], int] = {}
    reopen_counts: dict[int, int] = {}
    sr_seen: set[int] = set()
    total_events = 0
    assign_related = 0

    for batch in pf.iter_batches(columns=["FIELD", "ACTION", "SR_ID", "ACTION_DATE"], batch_size=500_000):
        block = batch.to_pandas()
        total_events += len(block)

        fields = (
            block["FIELD"]
            .astype("string")
            .fillna("NONE")
            .str.upper()
            .str.strip()
            .replace("", "NONE")
        )
        assign_related += int(fields.str.contains("ASSIGN|GROUP|DESK|OWNER", regex=True).sum())
        vc_field = fields.value_counts(dropna=False)
        for key, val in vc_field.items():
            field_counts[str(key)] = field_counts.get(str(key), 0) + int(val)

        actions = (
            block["ACTION"]
            .astype("string")
            .fillna("NONE")
            .str.upper()
            .str.strip()
            .replace("", "NONE")
        )
        vc_action = actions.value_counts(dropna=False)
        for key, val in vc_action.items():
            action_counts[str(key)] = action_counts.get(str(key), 0) + int(val)

        # Reopen heuristic from notebook cell 39 context (cell 38+39):
        # FIELD mentions REOPEN, or status change with reopen action, or action mentions REOPEN.
        sr_series = pd.to_numeric(block["SR_ID"], errors="coerce")
        reopen_mask = (
            fields.str.contains("REOPEN", na=False) |
            (fields.str.contains("STATUS", na=False) & actions.str.contains("REOPEN", na=False)) |
            actions.str.contains("REOPEN", na=False)
        )
        reopen_sr = sr_series[reopen_mask & sr_series.notna()].astype("int64")
        vc_reopen = reopen_sr.value_counts()
        for key, val in vc_reopen.items():
            reopen_counts[int(key)] = reopen_counts.get(int(key), 0) + int(val)

        if "ACTION_DATE" in block.columns:
            weeks = pd.to_datetime(block["ACTION_DATE"], errors="coerce").dt.to_period("W").dt.start_time
            wk = pd.DataFrame({"WEEK": weeks, "FIELD_U": fields}).dropna(subset=["WEEK"])
            if not wk.empty:
                vc_week = wk.groupby(["WEEK", "FIELD_U"]).size()
                for (week, field), val in vc_week.items():
                    key = (pd.Timestamp(week), str(field))
                    weekly_field_counts[key] = weekly_field_counts.get(key, 0) + int(val)

        sr_vals = sr_series.dropna().astype("int64").unique().tolist()
        sr_seen.update(sr_vals)

    top_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
    top_fields_df = pd.DataFrame(top_fields, columns=["FIELD", "count"])

    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_actions_df = pd.DataFrame(top_actions, columns=["ACTION", "count"])

    close_count = action_counts.get("CLOSE", 0)
    metrics = {
        "total_events": int(total_events),
        "unique_sr": int(len(sr_seen)),
        "assign_related_rate": float(assign_related / total_events) if total_events else np.nan,
        "close_action_rate": float(close_count / total_events) if total_events else np.nan,
    }
    weekly_df = pd.DataFrame(
        [{"WEEK": k[0], "FIELD_U": k[1], "n": v} for k, v in weekly_field_counts.items()]
    )
    if not weekly_df.empty:
        weekly_df = weekly_df.sort_values("WEEK").reset_index(drop=True)

    reopen_df = pd.DataFrame(
        [{"SR_ID": k, "n_reopens": v} for k, v in reopen_counts.items()]
    )
    if not reopen_df.empty:
        reopen_df["SR_ID"] = pd.to_numeric(reopen_df["SR_ID"], errors="coerce").astype("Int64")
        reopen_df["n_reopens"] = pd.to_numeric(reopen_df["n_reopens"], errors="coerce").fillna(0).astype(int)

    return top_fields_df, top_actions_df, weekly_df, reopen_df, metrics


def build_reopen_distribution(sr_handoffs: pd.DataFrame, reopen_df: pd.DataFrame, max_reopens: int = 10) -> pd.DataFrame:
    if sr_handoffs.empty or "SR_ID" not in sr_handoffs.columns:
        return pd.DataFrame()

    all_sr = pd.to_numeric(sr_handoffs["SR_ID"], errors="coerce").dropna().astype("int64").unique()
    if len(all_sr) == 0:
        return pd.DataFrame()

    base = pd.DataFrame({"SR_ID": all_sr})
    if reopen_df is not None and not reopen_df.empty:
        merged = base.merge(reopen_df[["SR_ID", "n_reopens"]], on="SR_ID", how="left")
    else:
        merged = base.copy()
        merged["n_reopens"] = 0

    merged["n_reopens"] = pd.to_numeric(merged["n_reopens"], errors="coerce").fillna(0).astype(int)
    dist = merged["n_reopens"].value_counts().sort_index()
    dist = dist[dist.index <= max_reopens]
    if dist.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "n_reopens": dist.index.astype(int),
        "tickets": dist.values.astype(int),
    })
    out["pct_tickets"] = out["tickets"] / out["tickets"].sum() * 100
    out = out.sort_values("n_reopens").reset_index(drop=True)
    return out


def plot_handoff_impact(impact: pd.DataFrame) -> go.Figure:
    plot_df = impact.copy()
    bucket_order = ["0", "1", "2-3", "4+"]
    plot_df["HANDOFF_BUCKET"] = pd.Categorical(
        plot_df["HANDOFF_BUCKET"].astype(str),
        categories=bucket_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("HANDOFF_BUCKET")

    has_overdue = "overdue_rate" in plot_df.columns and plot_df["overdue_rate"].notna().all()
    rate_col = "overdue_rate" if has_overdue else "open_rate"
    rate_title = "<b>Overdue SR rate (ASOF)</b>" if has_overdue else "<b>Open SR rate</b>"

    close_p90_raw = pd.to_numeric(plot_df["close_p90"], errors="coerce")
    close_p90_txt = close_p90_raw.apply(lambda v: f"{v:.1f} d" if pd.notna(v) else "n/a")
    close_fallback = max(0.5, float(close_p90_raw.dropna().max()) * 0.03) if close_p90_raw.notna().any() else 0.5
    close_p90_x = close_p90_raw.fillna(close_fallback)
    close_p90_colors = np.where(close_p90_raw.notna(), _COLOR_A, "#c9d8d2")

    rate_raw = pd.to_numeric(plot_df[rate_col], errors="coerce")
    rate_txt = rate_raw.apply(fmt_pct)
    rate_fallback = max(0.01, float(rate_raw.dropna().max()) * 0.1) if rate_raw.notna().any() else 0.01
    rate_x = rate_raw.fillna(rate_fallback)
    rate_colors = np.where(rate_raw.notna(), _BNP_DARK, "#c9d8d2")

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        column_titles=["<b>P90 close time (days)</b>", rate_title],
    )

    fig.add_trace(
        go.Bar(
            y=plot_df["HANDOFF_BUCKET"].astype(str),
            x=close_p90_x,
            orientation="h",
            marker_color=close_p90_colors,
            text=close_p90_txt,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Bucket %{y}<br>P90 close time: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["HANDOFF_BUCKET"].astype(str),
            x=rate_x,
            orientation="h",
            marker_color=rate_colors,
            text=rate_txt,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Bucket %{y}<br>Overdue SR rate: %{text}<extra></extra>"
                if has_overdue else
                "Bucket %{y}<br>Open SR rate: %{text}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=bucket_order,
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Impact of handoffs by SR bucket",
        height=max(380, len(plot_df) * 80),
        showlegend=False,
        margin=dict(l=0, r=95, t=70, b=20),
    )
    return fig


def plot_handoff_transitions(transitions: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top_trans = transitions.sort_values("n", ascending=False).head(top_n).copy()
    if top_trans.empty:
        return go.Figure()
    top_trans["EDGE"] = top_trans["FROM"].astype(str) + " -> " + top_trans["TO"].astype(str)
    top_trans = top_trans.sort_values("n", ascending=True)

    fig = px.bar(
        top_trans,
        x="n",
        y="EDGE",
        orientation="h",
        text="n",
        title=f"Top {len(top_trans)} group-to-group transitions",
        template=_TEMPLATE,
        color_discrete_sequence=[_COLOR_A],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(showgrid=False, showticklabels=False, title="")
    fig.update_yaxes(title="")
    fig.update_layout(
        height=max(420, len(top_trans) * 32),
        margin=dict(l=0, r=90, t=65, b=20),
    )
    return fig


def plot_history_weekly_fields(weekly_df: pd.DataFrame, top_fields_df: pd.DataFrame, top_k: int = 6) -> go.Figure:
    if weekly_df.empty:
        return go.Figure()

    keep_fields = top_fields_df["FIELD"].head(top_k).tolist() if not top_fields_df.empty else []
    data = weekly_df.copy()
    data["FIELD_GRP"] = np.where(data["FIELD_U"].isin(keep_fields), data["FIELD_U"], "OTHER")
    data = (
        data.groupby(["WEEK", "FIELD_GRP"], as_index=False)["n"]
        .sum()
        .sort_values("WEEK")
    )
    present_fields = data["FIELD_GRP"].dropna().astype(str).unique().tolist()
    ordered_fields = [f for f in keep_fields if f in present_fields]
    ordered_fields += [f for f in present_fields if f not in ordered_fields and f != "OTHER"]
    if "OTHER" in present_fields:
        ordered_fields.append("OTHER")

    contrast_palette = [
        "#0f5b46",
        "#167a5f",
        "#2f8f74",
        "#00a676",
        "#0d9488",
        "#0369a1",
        "#2563eb",
        "#7c3aed",
        "#db2777",
        "#f97316",
        "#ca8a04",
        "#475569",
    ]
    color_map = {field: contrast_palette[i % len(contrast_palette)] for i, field in enumerate(ordered_fields)}
    if "OTHER" in color_map:
        color_map["OTHER"] = "#1f2937"

    fig = px.area(
        data,
        x="WEEK",
        y="n",
        color="FIELD_GRP",
        category_orders={"FIELD_GRP": ordered_fields},
        color_discrete_map=color_map,
        title=f"History_SR — weekly change events (top {top_k} fields + OTHER)",
        template=_TEMPLATE,
    )
    fig.update_layout(height=450, margin=dict(l=0, r=20, t=65, b=20))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Events")
    return fig


def plot_reopen_distribution(dist_df: pd.DataFrame) -> go.Figure:
    if dist_df.empty:
        return go.Figure()

    plot_df = dist_df.copy()
    plot_df["bucket"] = plot_df["n_reopens"].astype(str)
    fig = px.bar(
        plot_df,
        x="bucket",
        y="pct_tickets",
        text=plot_df["pct_tickets"].map(lambda v: f"{v:.2f}%"),
        title="Distribution % — number of reopens per ticket",
        template=_TEMPLATE,
        color_discrete_sequence=[_COLOR_B],
    )
    fig.update_traces(cliponaxis=False)
    fig.update_layout(height=360, margin=dict(l=0, r=20, t=65, b=20))
    fig.update_xaxes(title="Number of reopens")
    fig.update_yaxes(title="% of tickets")
    return fig


def weekly_ts(df: pd.DataFrame, clip_q: float | None = None) -> pd.DataFrame:
    if "CREATIONDATE" not in df.columns and "CLOSINGDATE" not in df.columns:
        return pd.DataFrame()

    created = pd.Series(dtype=float)
    closed = pd.Series(dtype=float)

    if "CREATIONDATE" in df.columns:
        created = (
            df.dropna(subset=["CREATIONDATE"])
            .set_index("CREATIONDATE")
            .resample("W")
            .size()
            .rename("created")
        )
    if "CLOSINGDATE" in df.columns:
        closed = (
            df.dropna(subset=["CLOSINGDATE"])
            .set_index("CLOSINGDATE")
            .resample("W")
            .size()
            .rename("closed")
        )

    ts = (
        pd.concat([created, closed], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "week"})
    )
    if ts.empty:
        return ts

    ts["backlog_change"] = ts["created"] - ts["closed"]
    ts["backlog_estimated"] = ts["backlog_change"].cumsum()

    if clip_q is not None:
        cap_created = ts["created"].quantile(clip_q)
        cap_closed = ts["closed"].quantile(clip_q)
        ts["created"] = ts["created"].clip(upper=cap_created)
        ts["closed"] = ts["closed"].clip(upper=cap_closed)
    return ts


def overdue_by_week_created(df: pd.DataFrame, col: str = "OVERDUE_FLAG_ASOF") -> pd.DataFrame:
    if "CREATIONDATE" not in df.columns or col not in df.columns:
        return pd.DataFrame()
    tmp = df.dropna(subset=["CREATIONDATE"]).set_index("CREATIONDATE").resample("W")[col].mean()
    return tmp.reset_index().rename(columns={"CREATIONDATE": "week", col: "overdue_rate"})


def reopen_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    count_candidates = [
        "REOPEN_COUNT",
        "REOPENING_COUNT",
        "N_REOPEN",
        "NB_REOPEN",
        "NUMBER_OF_REOPENINGS",
    ]
    reopen_col = next((col for col in count_candidates if col in df.columns), None)

    source = None
    if reopen_col:
        reopen_values = pd.to_numeric(df[reopen_col], errors="coerce")
        source = reopen_col
    elif "IS_REOPENED" in df.columns:
        reopen_values = pd.to_numeric(df["IS_REOPENED"], errors="coerce")
        if reopen_values.isna().all():
            reopen_values = (
                df["IS_REOPENED"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["1", "true", "yes", "y"])
                .astype(float)
            )
        source = "IS_REOPENED (0/1)"
    elif "REOPEN_DATE" in df.columns:
        reopen_values = df["REOPEN_DATE"].notna().astype(float)
        source = "REOPEN_DATE (0/1)"
    else:
        return pd.DataFrame(), None

    work = pd.DataFrame({"reopen_count": reopen_values.fillna(0).clip(lower=0).round().astype(int)})
    id_col = "ID" if "ID" in df.columns else ("SR_ID" if "SR_ID" in df.columns else None)
    if id_col is not None:
        work["ticket_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        work = work.dropna(subset=["ticket_id"])
        if work.empty:
            return pd.DataFrame(), source
        count_values = work.groupby("ticket_id", as_index=False)["reopen_count"].max()["reopen_count"]
    else:
        count_values = work["reopen_count"]

    dist = (
        count_values.value_counts()
        .sort_index()
        .rename_axis("reopen_count")
        .reset_index(name="tickets")
    )
    if dist.empty:
        return dist, source

    dist["share"] = dist["tickets"] / dist["tickets"].sum()
    max_bucket = 10
    overflow = dist[dist["reopen_count"] >= max_bucket]
    if not overflow.empty and len(dist) > max_bucket:
        dist = dist[dist["reopen_count"] < max_bucket]
        dist = pd.concat(
            [
                dist,
                pd.DataFrame(
                    {
                        "reopen_count": [max_bucket],
                        "tickets": [int(overflow["tickets"].sum())],
                        "share": [float(overflow["share"].sum())],
                    }
                ),
            ],
            ignore_index=True,
        )
        dist["reopen_count_label"] = dist["reopen_count"].astype(str)
        dist.loc[dist["reopen_count"] == max_bucket, "reopen_count_label"] = f"{max_bucket}+"
    else:
        dist["reopen_count_label"] = dist["reopen_count"].astype(str)
    return dist, source


def activity_resolution_base(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["CREATIONDATE", "CLOSINGDATE", "CATEGORY_NAME"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    base = df[needed].dropna(subset=needed).copy()
    if base.empty:
        return base

    base["resolve_days"] = (base["CLOSINGDATE"] - base["CREATIONDATE"]).dt.total_seconds() / 86400
    base = base[base["resolve_days"].notna()]
    if base.empty:
        return base

    upper_bound = base["resolve_days"].quantile(0.95)
    return base[(base["resolve_days"] >= 0) & (base["resolve_days"] <= upper_bound)].copy()


def activity_top_resolution(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    base = activity_resolution_base(df)
    if base.empty:
        return pd.DataFrame()

    agg = (
        base.groupby("CATEGORY_NAME")
        .agg(avg_days=("resolve_days", "mean"), count=("resolve_days", "size"))
        .sort_values("avg_days", ascending=False)
        .head(top_n)
        .reset_index()
    )
    agg["display_name"] = agg["CATEGORY_NAME"].map(wrap_label)
    return agg


def activity_top_quick_tasks(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if "IS_QUICK_TASK" not in df.columns or "CATEGORY_NAME" not in df.columns:
        return pd.DataFrame()

    quick_flag = pd.to_numeric(df["IS_QUICK_TASK"], errors="coerce").fillna(0)
    top = (
        df.loc[quick_flag == 1]
        .dropna(subset=["CATEGORY_NAME"])
        .groupby("CATEGORY_NAME")
        .size()
        .rename("quick_count")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top["display_name"] = top["CATEGORY_NAME"].map(wrap_label)
    return top


def activity_ml_loss(df: pd.DataFrame, sla_quantile: float = 0.95, top_n: int = 20) -> tuple[pd.DataFrame, dict, str | None]:
    base = activity_resolution_base(df)
    if base.empty:
        return pd.DataFrame(), {}, "Not enough data to compute the SLA visualization."

    if len(base) < 20:
        return pd.DataFrame(), {}, "SLA model requires at least 20 valid rows."

    sla_map = (
        base.groupby("CATEGORY_NAME")["resolve_days"]
        .quantile(sla_quantile)
        .rename("sla_pred")
    )
    base = base.join(sla_map, on="CATEGORY_NAME")
    base["lost_days_ml"] = (base["resolve_days"] - base["sla_pred"]).clip(lower=0)
    base["breached"] = (base["lost_days_ml"] > 0).astype(int)

    metrics = {
        "mae": float(base["lost_days_ml"].mean()),
        "breach_rate": float(base["breached"].mean()),
        "total_lost": float(base["lost_days_ml"].sum()),
        "n_activities": len(base),
        "n_categories": base["CATEGORY_NAME"].nunique(),
    }

    loss = (
        base.groupby("CATEGORY_NAME")
        .agg(
            count=("resolve_days", "size"),
            sla_pred=("sla_pred", "median"),
            total_lost=("lost_days_ml", "sum"),
            breach_rate=("breached", "mean"),
        )
        .sort_values("total_lost", ascending=False)
        .head(top_n)
        .reset_index()
    )
    total_lost = loss["total_lost"].sum()
    loss["share_lost"] = np.where(total_lost > 0, loss["total_lost"] / total_lost * 100, 0.0)
    loss["display_name"] = loss["CATEGORY_NAME"].map(wrap_label)
    return loss, metrics, None


def plot_activity_resolution(agg: pd.DataFrame):
    n = len(agg)
    max_count = agg["count"].max() or 1
    max_days = agg["avg_days"].max() or 1
    norm_count = agg["count"] / max_count
    norm_days = agg["avg_days"] / max_days

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=agg["CATEGORY_NAME"],
        x=norm_count,
        orientation="h",
        name="# Actions",
        marker_color=_COLOR_A,
        text=agg["count"].apply(lambda v: f"{int(v):,}"),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br># Actions: %{text}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=agg["CATEGORY_NAME"],
        x=norm_days,
        orientation="h",
        name="Avg resolution time",
        marker_color=_COLOR_B,
        text=agg["avg_days"].apply(lambda v: f"{v:.1f} d"),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Avg resolution time: %{text}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Top {n} categories — action volume & resolution time",
            font=dict(size=16),
        ),
        template=_TEMPLATE,
        height=max(520, n * 54),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.04,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        margin=dict(l=0, r=110, t=80, b=70),
        font=dict(size=12),
    )
    return fig


def plot_activity_quick_tasks(top_quick: pd.DataFrame):
    n = len(top_quick)
    fig = px.bar(
        top_quick,
        y="CATEGORY_NAME",
        x="quick_count",
        orientation="h",
        text=top_quick["quick_count"].apply(lambda v: f"{int(v):,}"),
        title="Top 20 categories — quick tasks",
        template=_TEMPLATE,
        color_discrete_sequence=[_COLOR_A],
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=12))
    fig.update_xaxes(title="", showticklabels=False, showgrid=False)
    fig.update_layout(
        title_font_size=16,
        height=max(520, n * 44),
        margin=dict(l=0, r=90, t=65, b=20),
        font=dict(size=12),
    )
    return fig


def plot_activity_sla_loss(loss: pd.DataFrame, quantile: float):
    n = len(loss)
    df_plot = loss.copy()
    df_plot["label"] = df_plot.apply(
        lambda r: f"{r['total_lost']:.0f} d · {r['share_lost']:.1f}%", axis=1
    )
    fig = px.bar(
        df_plot,
        y="CATEGORY_NAME",
        x="total_lost",
        orientation="h",
        text="label",
        color="share_lost",
        color_continuous_scale=[[0, _BNP_TINT], [0.5, _COLOR_B], [1, _BNP_DARK]],
        title=f"Top categories — days lost beyond SLA P{int(quantile * 100)}",
        template=_TEMPLATE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=12))
    fig.update_xaxes(title="", showticklabels=False, showgrid=False)
    fig.update_layout(
        title_font_size=16,
        height=max(520, n * 44),
        margin=dict(l=0, r=160, t=65, b=20),
        coloraxis_showscale=False,
        font=dict(size=12),
    )
    return fig


def activity_overdue_table(df: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now()
    needed = ["CREATIONDATE"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    open_mask = df["CLOSINGDATE"].isna() if "CLOSINGDATE" in df.columns else pd.Series(True, index=df.index)
    base = df.loc[open_mask].copy()
    if base.empty:
        return pd.DataFrame()

    if "DUE_DATE" in base.columns:
        base["due"] = pd.to_datetime(base["DUE_DATE"], errors="coerce")
    else:
        base["due"] = base["CREATIONDATE"] + pd.Timedelta(days=5)

    overdue = base[base["due"] < now].copy()
    if overdue.empty:
        return pd.DataFrame()

    overdue["overdue_days"] = (now - overdue["due"]).dt.total_seconds() / 86400
    overdue["overdue_days"] = overdue["overdue_days"].round(1)

    keep_cols = [c for c in ["ID", "SR_ID", "CATEGORY_NAME", "CREATIONDATE", "due", "overdue_days"] if c in overdue.columns]
    return overdue[keep_cols].sort_values("overdue_days", ascending=False).reset_index(drop=True)


def plot_overdue_by_category(overdue_df: pd.DataFrame) -> go.Figure:
    if "CATEGORY_NAME" not in overdue_df.columns:
        return go.Figure()

    agg = (
        overdue_df.groupby("CATEGORY_NAME")
        .agg(nb_overdue=("overdue_days", "count"), total_overdue=("overdue_days", "sum"))
        .sort_values("total_overdue", ascending=False)
        .head(20)
        .reset_index()
    )

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_titles=["<b># Overdue activities</b>", "<b>Total overdue days</b>"],
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Bar(
            y=agg["CATEGORY_NAME"], x=agg["nb_overdue"], orientation="h",
            name="# Overdue", marker_color=_BNP_MID,
            text=agg["nb_overdue"].apply(lambda v: str(int(v))),
            textposition="outside", cliponaxis=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=agg["CATEGORY_NAME"], x=agg["total_overdue"], orientation="h",
            name="Overdue days", marker_color=_BNP_DARK,
            text=agg["total_overdue"].apply(lambda v: f"{v:.0f} d"),
            textposition="outside", cliponaxis=False,
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_layout(
        title=dict(text="Top 20 categories — open overdue activities", font=dict(size=16, color=_BNP_DARK)),
        template=_TEMPLATE,
        height=max(520, len(agg) * 44),
        showlegend=False,
        margin=dict(l=0, r=100, t=80, b=20),
        font=dict(size=12),
    )
    return fig


def render_sr_tab(path: str, start_year: int, clip_q: float | None):
    st.subheader("Service Requests")

    try:
        df = load_parquet(path)
    except FileNotFoundError:
        st.error(f"File not found: `{path}`")
        return
    except Exception as exc:
        st.error(f"Cannot load `{path}`: {exc}")
        return

    for col in ["CREATIONDATE", "CLOSINGDATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "CREATIONDATE" in df.columns:
        df = df[df["CREATIONDATE"] >= pd.Timestamp(f"{start_year}-01-01")]
    else:
        st.warning("Column `CREATIONDATE` is missing from SR data.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        cat_vals = sorted(df["CATEGORY_NAME"].dropna().unique().tolist()) if "CATEGORY_NAME" in df.columns else []
        cat_sel = st.multiselect("Category", cat_vals, default=[], key="sr_category")
    with col_b:
        prio_vals = sorted(df["PRIORITY_ID"].dropna().unique().tolist()) if "PRIORITY_ID" in df.columns else []
        prio_sel = st.multiselect("Priority", prio_vals, default=[], key="sr_priority")
    with col_c:
        status_vals = sorted(df["STATUS_ID"].dropna().unique().tolist()) if "STATUS_ID" in df.columns else []
        status_sel = st.multiselect("Status", status_vals, default=[], key="sr_status")
    with col_d:
        desk_vals = sorted(df["JUR_DESK_ID"].dropna().unique().tolist()) if "JUR_DESK_ID" in df.columns else []
        desk_sel = st.multiselect("Desk", desk_vals, default=[], key="sr_desk")

    mask = pd.Series(True, index=df.index)
    if cat_sel and "CATEGORY_NAME" in df.columns:
        mask &= df["CATEGORY_NAME"].isin(cat_sel)
    if prio_sel and "PRIORITY_ID" in df.columns:
        mask &= df["PRIORITY_ID"].isin(prio_sel)
    if status_sel and "STATUS_ID" in df.columns:
        mask &= df["STATUS_ID"].isin(status_sel)
    if desk_sel and "JUR_DESK_ID" in df.columns:
        mask &= df["JUR_DESK_ID"].isin(desk_sel)
    df_f = df.loc[mask].copy()

    if "IS_CLOSED" in df_f.columns:
        df_f["IS_CLOSED"] = pd.to_numeric(df_f["IS_CLOSED"], errors="coerce")
    else:
        df_f["IS_CLOSED"] = df_f["CLOSINGDATE"].notna().astype(float)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Tickets", f"{len(df_f):,}".replace(",", " "))
    kpi2.metric("Closed rate", fmt_pct(safe_mean(df_f["IS_CLOSED"])))
    kpi3.metric("Overdue rate", fmt_pct(safe_mean(df_f["OVERDUE_FLAG_ASOF"])) if "OVERDUE_FLAG_ASOF" in df_f.columns else "n/a")
    kpi4.metric("Ack on-time", fmt_pct(safe_mean(df_f["ACK_ON_TIME"])) if "ACK_ON_TIME" in df_f.columns else "n/a")
    kpi5.metric("1st response on-time", fmt_pct(safe_mean(df_f["FR_ON_TIME"])) if "FR_ON_TIME" in df_f.columns else "n/a")

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        ts = weekly_ts(df_f, clip_q=clip_q)
        if ts.empty:
            st.info("No weekly data to display.")
        else:
            fig = px.line(
                ts,
                x="week",
                y=["created", "closed"],
                title="Created vs Closed (weekly)",
                template=_TEMPLATE,
                color_discrete_sequence=[_BNP_PRIMARY, _BNP_SOFT],
            )
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(
                ts,
                x="week",
                y="backlog_estimated",
                title="Estimated backlog (cumulative created - closed)",
                template=_TEMPLATE,
                color_discrete_sequence=[_BNP_DARK],
            )
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        #ow = overdue_by_week_created(df_f, col="OVERDUE_FLAG_ASOF")
        #if not ow.empty:
        #    fig3 = px.line(ow, x="week", y="overdue_rate", title="Overdue rate by creation week")
        #    st.plotly_chart(fig3, use_container_width=True)

        if "CATEGORY_NAME" in df_f.columns and "OVERDUE_FLAG_ASOF" in df_f.columns:
            id_col = "ID" if "ID" in df_f.columns else "CATEGORY_NAME"
            by_cat = (
                df_f.groupby("CATEGORY_NAME")
                .agg(n=(id_col, "count"), overdue_rate=("OVERDUE_FLAG_ASOF", "mean"))
                .reset_index()
            )
            by_cat["overdue_volume"] = by_cat["n"] * by_cat["overdue_rate"]
            top = by_cat.sort_values("overdue_volume", ascending=False).head(15)
            fig4 = px.bar(
                top,
                x="overdue_volume",
                y=top["CATEGORY_NAME"].astype(str),
                orientation="h",
                title="Top 15 categories by overdue volume",
                template=_TEMPLATE,
                color_discrete_sequence=[_BNP_MID],
            )
            st.plotly_chart(fig4, use_container_width=True)

        reopen_dist, reopen_source = reopen_distribution(df_f)
        if not reopen_dist.empty:
            fig5 = px.bar(
                reopen_dist,
                x="reopen_count_label",
                y="tickets",
                text=reopen_dist["share"].map(lambda v: f"{v:.1%}"),
                title="Ticket reopen count distribution",
                template=_TEMPLATE,
                color_discrete_sequence=[_COLOR_A],
            )
            fig5.update_traces(textposition="outside", cliponaxis=False)
            fig5.update_xaxes(title="Number of reopenings", type="category")
            fig5.update_yaxes(title="Ticket count")
            fig5.update_layout(margin=dict(l=0, r=30, t=60, b=10))
            st.plotly_chart(fig5, use_container_width=True)
            

    st.divider()
    st.subheader("Sample rows (drill-down)")
    show_cols = [
        c
        for c in [
            "ID",
            "SRNUMBER",
            "CATEGORY_NAME",
            "PRIORITY_ID",
            "STATUS_ID",
            "CREATIONDATE",
            "CLOSINGDATE",
            "DUE_DATE",
            "OVERDUE_FLAG_ASOF",
            "OVERDUE_DAYS_ASOF",
            "CLOSE_DELAY_D",
            "ACK_DELAY_H",
            "FR_DELAY_H",
        ]
        if c in df_f.columns
    ]
    if show_cols:
        st.dataframe(df_f[show_cols].head(500))
    else:
        st.dataframe(df_f.head(500))


def render_activity_tab(path: str):
    st.subheader("Activity")
    st.caption(f"Source: `{path}`")

    try:
        df = load_parquet(path)
    except FileNotFoundError:
        st.error(f"File not found: `{path}`")
        return
    except Exception as exc:
        st.error(f"Cannot load `{path}`: {exc}")
        return

    for col in ["CREATIONDATE", "CLOSINGDATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    import datetime as _dt

    _date_min = df["CREATIONDATE"].min().date() if "CREATIONDATE" in df.columns and df["CREATIONDATE"].notna().any() else _dt.date(2020, 1, 1)
    _date_max = df["CREATIONDATE"].max().date() if "CREATIONDATE" in df.columns and df["CREATIONDATE"].notna().any() else _dt.date.today()

    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            cat_vals = sorted(df["CATEGORY_NAME"].dropna().astype(str).unique().tolist()) if "CATEGORY_NAME" in df.columns else []
            cat_sel = st.multiselect("Category", cat_vals, default=[], key="activity_category")
        with f2:
            date_range = st.date_input(
                "Creation date range",
                value=(_date_min, _date_max),
                min_value=_date_min,
                max_value=_date_max,
                key="activity_date_range",
            )

        f3, f4, f5 = st.columns(3)
        with f3:
            status_opts = ["All", "Open only", "Closed only"]
            status_sel = st.radio("Status", status_opts, index=0, key="activity_status", horizontal=True)
        with f4:
            qt_opts = ["All", "Quick tasks", "Non quick tasks"]
            qt_sel = st.radio("Quick task", qt_opts, index=0, key="activity_qt", horizontal=True)
        with f5:
            sla_quantile = st.select_slider(
                "SLA percentile",
                options=[0.80, 0.85, 0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda v: f"P{int(v*100)}",
                key="activity_sla_q",
            )

    top_n = st.slider("Top N categories", min_value=5, max_value=40, value=20, step=1, key="activity_top_n")

    df_f = df.copy()
    if cat_sel and "CATEGORY_NAME" in df_f.columns:
        df_f = df_f[df_f["CATEGORY_NAME"].astype(str).isin(cat_sel)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and "CREATIONDATE" in df_f.columns:
        d_from, d_to = date_range
        df_f = df_f[
            (df_f["CREATIONDATE"].dt.date >= d_from) &
            (df_f["CREATIONDATE"].dt.date <= d_to)
        ]
    if status_sel == "Open only" and "CLOSINGDATE" in df_f.columns:
        df_f = df_f[df_f["CLOSINGDATE"].isna()]
    elif status_sel == "Closed only" and "CLOSINGDATE" in df_f.columns:
        df_f = df_f[df_f["CLOSINGDATE"].notna()]
    if "IS_QUICK_TASK" in df_f.columns:
        _qt_flag = pd.to_numeric(df_f["IS_QUICK_TASK"], errors="coerce").fillna(0)
        if qt_sel == "Quick tasks":
            df_f = df_f[_qt_flag == 1]
        elif qt_sel == "Non quick tasks":
            df_f = df_f[_qt_flag != 1]

    is_closed = df_f["CLOSINGDATE"].notna() if "CLOSINGDATE" in df_f.columns else pd.Series(False, index=df_f.index)
    quick_rate = np.nan
    if "IS_QUICK_TASK" in df_f.columns and len(df_f):
        quick_rate = float((pd.to_numeric(df_f["IS_QUICK_TASK"], errors="coerce").fillna(0) == 1).mean())

    base_resolution = activity_resolution_base(df_f)
    avg_resolution_days = safe_mean(base_resolution["resolve_days"]) if not base_resolution.empty else np.nan

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Activities", f"{len(df_f):,}".replace(",", " "))
    kpi2.metric("Closed rate", fmt_pct(float(is_closed.mean())) if len(df_f) else "n/a")
    kpi3.metric("Quick task rate", fmt_pct(quick_rate))
    kpi4.metric("Avg resolution time", fmt_days(avg_resolution_days))

    st.divider()

    with st.spinner("Computing SLA model..."):
        loss_df, ml_metrics, error_msg = activity_ml_loss(df_f, sla_quantile=sla_quantile, top_n=top_n)

    st.markdown(f"#### SLA Breach Prediction — Quantile Regression (P{int(sla_quantile * 100)})")
    if error_msg:
        st.info(error_msg)
    elif loss_df.empty:
        st.info("Chart unavailable: no SLA result.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (avg breach)", f"{ml_metrics['mae']:.2f} d", help="Mean Absolute Error — average days lost per activity beyond the predicted SLA threshold")
        m2.metric("Breach rate", fmt_pct(ml_metrics["breach_rate"]), help="Share of activities that exceeded their predicted SLA")
        m3.metric("Total days lost", f"{ml_metrics['total_lost']:,.0f} d", help="Cumulative days lost across all SLA breaches")
        m4.metric("Categories modelled", int(ml_metrics["n_categories"]), help="Number of distinct categories used to fit the quantile model")

        sla_left, sla_right = st.columns([3, 2])
        with sla_left:
            st.plotly_chart(plot_activity_sla_loss(loss_df, quantile=sla_quantile), use_container_width=True)
        with sla_right:
            details = loss_df[["CATEGORY_NAME", "count", "sla_pred", "total_lost", "share_lost", "breach_rate"]].copy()
            details = details.rename(columns={
                "CATEGORY_NAME": "Category",
                "count": "# Activities",
                "sla_pred": f"SLA P{int(sla_quantile*100)} (d)",
                "total_lost": "Lost days",
                "share_lost": "Share (%)",
                "breach_rate": "Breach rate",
            })
            details[f"SLA P{int(sla_quantile*100)} (d)"] = details[f"SLA P{int(sla_quantile*100)} (d)"].round(1)
            details["Lost days"] = details["Lost days"].round(1)
            details["Share (%)"] = details["Share (%)"].round(1)
            details["Breach rate"] = details["Breach rate"].apply(fmt_pct)
            st.dataframe(details, use_container_width=True, hide_index=True, height=min(620, 45 + len(details) * 35))

    st.divider()

    res_col, qt_col = st.columns(2)
    with res_col:
        top_resolution = activity_top_resolution(df_f, top_n=top_n)
        st.markdown("#### Action volume & resolution time")
        if top_resolution.empty:
            st.info("Chart unavailable: required columns missing or insufficient data.")
        else:
            st.plotly_chart(plot_activity_resolution(top_resolution), use_container_width=True)
    with qt_col:
        top_quick = activity_top_quick_tasks(df_f, top_n=top_n)
        st.markdown("#### Quick tasks by category")
        if top_quick.empty:
            st.info("Chart unavailable: column `IS_QUICK_TASK` is missing or empty.")
        else:
            st.plotly_chart(plot_activity_quick_tasks(top_quick), use_container_width=True)

    st.divider()

    overdue_df = activity_overdue_table(df_f)
    n_overdue = len(overdue_df)

    if n_overdue == 0:
        st.success("No open overdue activities detected.")
    else:
        st.error(f"**{n_overdue:,} open overdue activity(ies)** — {overdue_df['overdue_days'].sum():.0f} cumulative overdue days")

        st.markdown("#### Overdue by category")
        st.caption("Click a bar to filter activities for that category.")
        overdue_event = st.plotly_chart(
            plot_overdue_by_category(overdue_df),
            use_container_width=True,
            on_select="rerun",
            key="overdue_chart",
        )

        selected_cat = None
        pts = (overdue_event or {}).get("selection", {}).get("points", [])
        if pts:
            selected_cat = pts[0].get("y")

        detail_df = overdue_df[overdue_df["CATEGORY_NAME"] == selected_cat] if selected_cat else overdue_df
        detail_label = f"Category: **{selected_cat}**" if selected_cat else f"All categories ({n_overdue} activities)"

        detail_cols = [c for c in ["ID", "SR_ID", "CATEGORY_NAME", "CREATIONDATE", "due", "overdue_days"] if c in detail_df.columns]
        detail_renamed = detail_df[detail_cols].rename(columns={
            "CATEGORY_NAME": "Category",
            "overdue_days": "Overdue (d)",
            "CREATIONDATE": "Created on",
            "due": "Due date",
        })

        st.markdown(f"##### Overdue activities — {detail_label}")
        if selected_cat:
            nb = len(detail_renamed)
            total_j = detail_df["overdue_days"].sum()
            st.info(f"{nb} activity(ies) — {total_j:.0f} cumulative overdue days")
        st.dataframe(detail_renamed.sort_values("Overdue (d)", ascending=False), use_container_width=True, hide_index=True)


def render_handoffs_history_tab(
    activity_graph_path: str,
    history_sr_path: str,
    network_html_path: str,
    sr_enriched_path: str,
):
    st.subheader("Handoffs & History SR")
    

    try:
        activity_df = load_parquet(activity_graph_path)
    except FileNotFoundError:
        st.error(f"File not found: `{activity_graph_path}`")
        return
    except Exception as exc:
        st.error(f"Cannot load `{activity_graph_path}`: {exc}")
        return

    sr_overdue_map = pd.DataFrame()
    overdue_available = False
    try:
        sr_overdue_map = load_sr_overdue_map(sr_enriched_path)
        overdue_available = not sr_overdue_map.empty
    except FileNotFoundError:
        st.warning(f"File not found for overdue join: `{sr_enriched_path}`. Falling back to open-rate proxy.")
    except Exception as exc:
        st.warning(f"Cannot read overdue data from `{sr_enriched_path}` ({exc}). Falling back to open-rate proxy.")

    sr_handoffs, impact, transitions, meta, handoff_error = prepare_handoff_data(
        activity_df,
        sr_overdue_map=sr_overdue_map,
    )
    if handoff_error:
        st.error(handoff_error)
        return

    try:
        with st.spinner("Computing History SR aggregates..."):
            fields_df, actions_df, weekly_df, reopen_df, hist_metrics = history_summary(history_sr_path)
    except FileNotFoundError:
        st.error(f"File not found: `{history_sr_path}`")
        return
    except Exception as exc:
        st.error(f"Cannot load `{history_sr_path}`: {exc}")
        return

    reopen_dist_df = build_reopen_distribution(sr_handoffs, reopen_df, max_reopens=10)

    handoff_sr = int((sr_handoffs["n_handoffs"] > 0).sum()) if len(sr_handoffs) else 0
    multi_group_sr = int((sr_handoffs["n_groups"] > 1).sum()) if len(sr_handoffs) else 0
    avg_handoffs = safe_mean(sr_handoffs["n_handoffs"]) if len(sr_handoffs) else np.nan
    p90_handoffs = safe_q90(sr_handoffs["n_handoffs"]) if len(sr_handoffs) else np.nan
    overdue_rate = safe_mean(sr_handoffs["OVERDUE_FLAG_ASOF"]) if overdue_available else np.nan

    top_transition_count = 0
    top_transition_label = "n/a"
    if not transitions.empty:
        top_row = transitions.sort_values("n", ascending=False).iloc[0]
        top_transition_count = int(top_row["n"])
        top_transition_label = f"{top_row['FROM']} -> {top_row['TO']}"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("SR analysed", f"{meta.get('n_sr', 0):,}".replace(",", " "))
    k2.metric("SR with handoff", fmt_pct(handoff_sr / meta["n_sr"] if meta.get("n_sr") else np.nan))
    if pd.notna(overdue_rate):
        k3.metric("Overdue rate (ASOF)", fmt_pct(overdue_rate))
    else:
        k3.metric("Multi-group SR", fmt_pct(multi_group_sr / meta["n_sr"] if meta.get("n_sr") else np.nan))
    k4.metric("Avg handoffs / SR", f"{avg_handoffs:.2f}" if pd.notna(avg_handoffs) else "n/a")
    k5.metric(
        "Top transition",
        f"{top_transition_count:,}".replace(",", " "),
        help=f"Most frequent flow: {top_transition_label}",
    )
    p90_label = f"{p90_handoffs:.1f}" if pd.notna(p90_handoffs) else "n/a"
    multi_group_label = fmt_pct(multi_group_sr / meta["n_sr"] if meta.get("n_sr") else np.nan)
    #st.caption(
    #    f"Computed with group column `{meta.get('group_col', 'n/a')}` and timeline column `{meta.get('time_col', 'n/a')}` · "
    #    f"P90 handoffs per SR: {p90_label} · Multi-group SR: {multi_group_label}"
    #)

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Handoff impact by bucket")
        if impact.empty:
            st.info("Impact chart unavailable.")
        else:
            st.plotly_chart(plot_handoff_impact(impact), use_container_width=True)
    with right:
        st.markdown("#### Reopens per ticket")
        if reopen_dist_df.empty:
            st.info("No reopen distribution available.")
        else:
            st.plotly_chart(plot_reopen_distribution(reopen_dist_df), use_container_width=True)

    st.divider()

    st.markdown("#### Transition view")
    tcol1, tcol2 = st.columns([2, 1])
    with tcol1:
        transition_view = st.radio(
            "Interchangeable view",
            options=["Top transitions (bar chart)", "Routing network (HTML)"],
            horizontal=True,
            key="handoff_transition_view",
        )
    with tcol2:
        top_n_trans = st.slider(
            "Top transitions",
            min_value=10,
            max_value=120,
            value=20,
            step=5,
            key="handoff_top_transitions",
        )

    if transition_view == "Top transitions (bar chart)":
        if transitions.empty:
            st.info("No handoff transitions found.")
        else:
            st.plotly_chart(plot_handoff_transitions(transitions, top_n=top_n_trans), use_container_width=True)
    else:
        try:
            html_content = load_html(network_html_path)
            html_content = html_content.replace('<script src="lib/bindings/utils.js"></script>', "")
            components.html(html_content, height=820, scrolling=True)
            st.caption(f"Network loaded from `{network_html_path}`")
        except FileNotFoundError:
            st.error(f"File not found: `{network_html_path}`")
        except Exception as exc:
            st.error(f"Cannot load `{network_html_path}`: {exc}")

    #if not transitions.empty:
    #    top_transitions = transitions.sort_values("n", ascending=False).head(top_n_trans).copy()
    #    top_transitions["Transition"] = top_transitions["FROM"].astype(str) + " -> " + top_transitions["TO"].astype(str)
    #    st.dataframe(
    #        top_transitions[["Transition", "n"]].rename(columns={"n": "Handoffs"}),
    #        use_container_width=True,
    #        hide_index=True,
    #    )

    st.divider()
    st.markdown("#### History SR changes")

    top_n_fields = st.slider(
        "Top fields in weekly view",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
        key="history_top_fields",
    )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("History events", f"{hist_metrics['total_events']:,}".replace(",", " "))
    h2.metric("SR impacted", f"{hist_metrics['unique_sr']:,}".replace(",", " "))
    h3.metric("Assign/group related", fmt_pct(hist_metrics["assign_related_rate"]))
    h4.metric("Close actions", fmt_pct(hist_metrics["close_action_rate"]))

    hist_left, hist_right = st.columns([2, 1])
    with hist_left:
        if weekly_df.empty:
            st.info("No weekly history data available (`ACTION_DATE` missing/empty).")
        else:
            st.plotly_chart(
                plot_history_weekly_fields(weekly_df, fields_df, top_k=top_n_fields),
                use_container_width=True,
            )
    with hist_right:
        if actions_df.empty:
            st.info("No history actions available.")
        else:
            top_actions = actions_df.head(10).sort_values("count", ascending=True)
            fig_actions = px.bar(
                top_actions,
                x="count",
                y="ACTION",
                orientation="h",
                title="Top actions",
                template=_TEMPLATE,
                color_discrete_sequence=[_COLOR_A],
            )
            fig_actions.update_traces(text=top_actions["count"], textposition="outside", cliponaxis=False)
            fig_actions.update_xaxes(showgrid=False, showticklabels=False, title="")
            fig_actions.update_yaxes(title="")
            fig_actions.update_layout(height=420, margin=dict(l=0, r=90, t=60, b=20))
            st.plotly_chart(fig_actions, use_container_width=True)


if Path(_BANNER_PATH).exists():
    st.image(_BANNER_PATH, use_container_width=True)

st.title("BNP Paribas Operations Intelligence Dashboard")
st.caption("SLA breach prediction powered by quantile regression · Activity, SR, handoffs & history analytics")

with st.sidebar:
    st.header("SR Data")
    sr_path = st.text_input(
        "Path to sr_enriched.parquet",
        value="https://drive.google.com/file/d/1nawUFaBJWXX3B7mSi0uaM0ltlKyc_kdr/view?usp=sharing",
    )

    st.header("SR Filters")
    start_year = st.selectbox("Start from year", options=[2024, 2025, 2026], index=1)
    clip_outliers = st.checkbox("Clip extreme weekly peaks (recommended)", value=True)
    clip_q = 0.995 if clip_outliers else None

    st.header("Handoffs & History Data")
    activity_path = st.text_input(
        "Path to activity.parquet",
        value="https://drive.google.com/file/d/1Ng4dQ3E5UfRd8DYt99PwuXbbew0h2MNx/view?usp=drive_link",
    )
    activity_graph_path = st.text_input(
        "Path to Activity_Jan_to_Sept_graph.parquet",
        value="https://drive.google.com/file/d/1nawUFaBJWXX3B7mSi0uaM0ltlKyc_kdr/view?usp=drive_link",
    )
    history_sr_path = st.text_input(
        "Path to History_SR_Jan_to_Sept.parquet",
        value="https://drive.google.com/file/d/1v5aJ6cFN-0-UnkJLm9hkERLVKqyWN5TW/view?usp=drive_link",
    )
    network_html_path = st.text_input(
        "Path to routing network HTML",
        value="https://drive.google.com/file/d/1FumhWE4zTzOBLzS2U8pABB3JZZOlR52z/view?usp=drive_link",
    )

tab_sr, tab_activity, tab_handoff_history = st.tabs(["SR", "Activity", "Handoffs & History SR"])
with tab_sr:
    render_sr_tab(sr_path, start_year, clip_q)
with tab_activity:
    render_activity_tab(activity_path)
with tab_handoff_history:
    render_handoffs_history_tab(activity_graph_path, history_sr_path, network_html_path, sr_path)
