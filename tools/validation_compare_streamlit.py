import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_RUN_DIR = Path(
    "/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/crrl/"
    "crrl_verl_pr/Qwen3-4B_CRRL_batch_1024_B200_dynamicsampling/validation_data"
)


def _short_prompt(text: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_problem(prompt: str) -> str:
    marker = "Solve the following math problem step by step."
    if marker in prompt:
        return prompt.split(marker, 1)[-1].strip()
    return prompt.strip()


def _extract_final_answer(output: str) -> str:
    matches = re.findall(r"Answer:\s*([^\n\r]+)", output or "", flags=re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def _strip_think(output: str) -> str:
    if not output:
        return ""
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


@st.cache_data(show_spinner=False)
def load_jsonl(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["input", "output", "gts", "pred"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in ["score", "reward"]:
        if col not in df:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["prompt_id"] = pd.factorize(df["input"])[0]
    df["sample_idx"] = df.groupby("input").cumcount()
    df["problem"] = df["input"].map(_extract_problem)
    df["prompt_short"] = df["problem"].map(_short_prompt)
    df["output_chars"] = df["output"].str.len()
    df["output_words"] = df["output"].str.split().map(len)
    df["visible_output"] = df["output"].map(_strip_think)
    df["visible_chars"] = df["visible_output"].str.len()
    df["final_answer"] = df["output"].map(_extract_final_answer)
    return df


def paired_frame(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    keep = [
        "input",
        "prompt_id",
        "sample_idx",
        "problem",
        "prompt_short",
        "output",
        "visible_output",
        "output_chars",
        "output_words",
        "visible_chars",
        "score",
        "reward",
        "gts",
        "pred",
        "final_answer",
        "_line_no",
    ]
    left = df_a[[c for c in keep if c in df_a]].copy()
    right = df_b[[c for c in keep if c in df_b]].copy()
    merged = left.merge(
        right,
        on=["input", "sample_idx"],
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    )
    merged["prompt_id"] = merged[f"prompt_id_{label_a}"]
    merged["problem"] = merged[f"problem_{label_a}"]
    merged["prompt_short"] = merged[f"prompt_short_{label_a}"]
    merged["gts"] = merged[f"gts_{label_a}"].where(
        merged[f"gts_{label_a}"].astype(str).str.len() > 0,
        merged.get(f"gts_{label_b}", ""),
    )
    merged["score_delta"] = merged[f"score_{label_b}"] - merged[f"score_{label_a}"]
    merged["chars_delta"] = merged[f"output_chars_{label_b}"] - merged[f"output_chars_{label_a}"]
    merged["words_delta"] = merged[f"output_words_{label_b}"] - merged[f"output_words_{label_a}"]
    merged["visible_chars_delta"] = merged[f"visible_chars_{label_b}"] - merged[f"visible_chars_{label_a}"]
    merged["shorter"] = merged["chars_delta"] < 0
    merged["score_changed"] = merged["score_delta"].fillna(0) != 0
    return merged


def metric_box(label: str, value: str) -> None:
    st.metric(label, value)


st.set_page_config(page_title="Validation Response Compare", layout="wide")
st.title("Validation Response Compare")

with st.sidebar:
    st.header("Files")
    default_a = str(DEFAULT_RUN_DIR / "0.jsonl")
    default_b = str(DEFAULT_RUN_DIR / "80.jsonl")
    path_a = st.text_input("Step A JSONL", default_a)
    path_b = st.text_input("Step B JSONL", default_b)
    label_a = st.text_input("Step A label", "0")
    label_b = st.text_input("Step B label", "80")

df_a = load_jsonl(path_a)
df_b = load_jsonl(path_b)

if df_a.empty or df_b.empty:
    st.error("One of the files is empty or could not be parsed.")
    st.stop()

pairs = paired_frame(df_a, df_b, label_a, label_b)
if pairs.empty:
    st.error("No matching input/sample_idx pairs found.")
    st.stop()

summary_cols = st.columns(6)
with summary_cols[0]:
    metric_box("Paired samples", f"{len(pairs):,}")
with summary_cols[1]:
    metric_box("Prompts", f"{pairs['input'].nunique():,}")
with summary_cols[2]:
    metric_box(f"Avg score {label_a}", f"{pairs[f'score_{label_a}'].mean():.3f}")
with summary_cols[3]:
    metric_box(f"Avg score {label_b}", f"{pairs[f'score_{label_b}'].mean():.3f}")
with summary_cols[4]:
    metric_box("Avg char delta", f"{pairs['chars_delta'].mean():+.0f}")
with summary_cols[5]:
    metric_box("Shorter ratio", f"{pairs['shorter'].mean():.1%}")

with st.sidebar:
    st.header("Filters")
    mode = st.selectbox(
        "Subset",
        [
            "All",
            "Step B shorter",
            "Step B much shorter (>1000 chars)",
            "Score improved",
            "Score regressed",
            "Score changed",
            "A wrong, B correct",
            "A correct, B wrong",
        ],
    )
    min_abs_char_delta = st.slider("Minimum abs char delta", 0, 8000, 0, 100)
    sample_idx_options = ["All"] + sorted(pairs["sample_idx"].unique().tolist())
    sample_idx = st.selectbox("Sample index", sample_idx_options)
    show_think = st.checkbox("Show full output including <think>", value=True)

filtered = pairs.copy()
if mode == "Step B shorter":
    filtered = filtered[filtered["chars_delta"] < 0]
elif mode == "Step B much shorter (>1000 chars)":
    filtered = filtered[filtered["chars_delta"] < -1000]
elif mode == "Score improved":
    filtered = filtered[filtered["score_delta"] > 0]
elif mode == "Score regressed":
    filtered = filtered[filtered["score_delta"] < 0]
elif mode == "Score changed":
    filtered = filtered[filtered["score_changed"]]
elif mode == "A wrong, B correct":
    filtered = filtered[(filtered[f"score_{label_a}"] <= 0) & (filtered[f"score_{label_b}"] > 0)]
elif mode == "A correct, B wrong":
    filtered = filtered[(filtered[f"score_{label_a}"] > 0) & (filtered[f"score_{label_b}"] <= 0)]

filtered = filtered[filtered["chars_delta"].abs() >= min_abs_char_delta]
if sample_idx != "All":
    filtered = filtered[filtered["sample_idx"] == int(sample_idx)]

prompt_rows = (
    filtered.groupby(["input", "prompt_short"], as_index=False)
    .agg(
        samples=("sample_idx", "count"),
        score_delta_mean=("score_delta", "mean"),
        chars_delta_mean=("chars_delta", "mean"),
        shorter_ratio=("shorter", "mean"),
    )
    .sort_values(["chars_delta_mean", "score_delta_mean"], ascending=[True, True])
)

st.subheader("Filtered Overview")
st.write(f"{len(filtered):,} paired samples after filters")
left_chart, right_chart = st.columns(2)
with left_chart:
    st.caption("Character delta distribution")
    st.bar_chart(filtered["chars_delta"].value_counts(bins=30).sort_index())
with right_chart:
    st.caption("Score delta distribution")
    st.bar_chart(filtered["score_delta"].fillna(0).value_counts().sort_index())

st.dataframe(
    prompt_rows.rename(
        columns={
            "prompt_short": "prompt",
            "score_delta_mean": "mean score delta",
            "chars_delta_mean": "mean char delta",
            "shorter_ratio": "shorter ratio",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if prompt_rows.empty:
    st.warning("No samples match the current filters.")
    st.stop()

prompt_label_to_input = {
    f"[{i}] dchars={row.chars_delta_mean:+.0f}, dscore={row.score_delta_mean:+.2f} | {row.prompt_short}": row.input
    for i, row in enumerate(prompt_rows.itertuples(), start=1)
}
selected_label = st.selectbox("Prompt", list(prompt_label_to_input.keys()))
selected_input = prompt_label_to_input[selected_label]
prompt_samples = filtered[filtered["input"] == selected_input].sort_values(["sample_idx"])

sample_labels = {
    (
        f"sample {int(row.sample_idx)} | "
        f"score {row[f'score_{label_a}']:.0f}->{row[f'score_{label_b}']:.0f} | "
        f"chars {row[f'output_chars_{label_a}']:.0f}->{row[f'output_chars_{label_b}']:.0f} "
        f"({row.chars_delta:+.0f})"
    ): idx
    for idx, row in prompt_samples.iterrows()
}
selected_sample_label = st.selectbox("Sample", list(sample_labels.keys()))
row = pairs.loc[sample_labels[selected_sample_label]]

st.subheader("Problem")
st.code(row["problem"], language="markdown")
st.write(f"Ground truth: `{row['gts']}`")

meta_a, meta_b, meta_delta = st.columns(3)
with meta_a:
    st.markdown(f"**Step {label_a}**")
    st.write(
        {
            "score": row.get(f"score_{label_a}"),
            "pred": row.get(f"pred_{label_a}"),
            "final_answer": row.get(f"final_answer_{label_a}"),
            "chars": row.get(f"output_chars_{label_a}"),
            "words": row.get(f"output_words_{label_a}"),
        }
    )
with meta_b:
    st.markdown(f"**Step {label_b}**")
    st.write(
        {
            "score": row.get(f"score_{label_b}"),
            "pred": row.get(f"pred_{label_b}"),
            "final_answer": row.get(f"final_answer_{label_b}"),
            "chars": row.get(f"output_chars_{label_b}"),
            "words": row.get(f"output_words_{label_b}"),
        }
    )
with meta_delta:
    st.markdown("**Delta**")
    st.write(
        {
            "score_delta": row.get("score_delta"),
            "chars_delta": row.get("chars_delta"),
            "words_delta": row.get("words_delta"),
            "visible_chars_delta": row.get("visible_chars_delta"),
        }
    )

out_col_a, out_col_b = st.columns(2)
output_key = "output" if show_think else "visible_output"
with out_col_a:
    st.markdown(f"### Step {label_a} Output")
    st.text_area(
        f"{label_a} output",
        row.get(f"{output_key}_{label_a}", ""),
        height=720,
        label_visibility="collapsed",
    )
with out_col_b:
    st.markdown(f"### Step {label_b} Output")
    st.text_area(
        f"{label_b} output",
        row.get(f"{output_key}_{label_b}", ""),
        height=720,
        label_visibility="collapsed",
    )

download_cols = [
    "prompt_id",
    "sample_idx",
    "gts",
    f"score_{label_a}",
    f"score_{label_b}",
    "score_delta",
    f"output_chars_{label_a}",
    f"output_chars_{label_b}",
    "chars_delta",
    f"pred_{label_a}",
    f"pred_{label_b}",
    "problem",
]
st.download_button(
    "Download filtered pairs CSV",
    filtered[[c for c in download_cols if c in filtered]].to_csv(index=False).encode("utf-8"),
    file_name=f"validation_compare_{label_a}_vs_{label_b}.csv",
    mime="text/csv",
)
