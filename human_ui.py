import streamlit as st
import os
import time
import pandas as pd
import random
from PIL import Image
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
CAPTCHA_FOLDER = "real_captcha_samples"
RESULTS_PATH = "results/human_results.csv"
MAX_TIME = 10  # max allowed time per CAPTCHA

# =========================
# LOAD CAPTCHA FILES
# =========================
if not os.path.exists(CAPTCHA_FOLDER):
    st.error("❌ CAPTCHA folder not found. Run main.py first.")
    st.stop()

files = [f for f in os.listdir(CAPTCHA_FOLDER) if f.endswith(".png")]

if len(files) == 0:
    st.error("❌ No CAPTCHA images found.")
    st.stop()

# =========================
# SESSION STATE INIT
# =========================
if "initialized" not in st.session_state:
    random.shuffle(files)

    st.session_state.files = files
    st.session_state.index = 0
    st.session_state.correct = 0
    st.session_state.total = 0
    st.session_state.start_time = None
    st.session_state.times = []
    st.session_state.records = []
    st.session_state.user = ""
    st.session_state.initialized = True

# =========================
# USER INPUT
# =========================
if st.session_state.user == "":
    st.title("👤 Enter User Details")
    user = st.text_input("Enter your name / ID:")
    if user.strip() == "":
        st.stop()
    st.session_state.user = user

# =========================
# TITLE
# =========================
st.title("🔐 CAPTCHA Human Evaluation System")

# Debug toggle
show_answer = st.checkbox("Show correct answer (Debug Mode)")

# =========================
# FINISH SCREEN
# =========================
if st.session_state.index >= len(st.session_state.files):

    if st.session_state.total > 0:
        accuracy = st.session_state.correct / st.session_state.total
        avg_time = sum([t for t in st.session_state.times if t is not None]) / len(st.session_state.times)

        st.success("✅ Evaluation Completed!")

        st.subheader("📊 Final Results")
        st.write(f"👤 User: {st.session_state.user}")
        st.write(f"• Accuracy: {accuracy:.4f}")
        st.write(f"• Avg Solve Time: {avg_time:.2f} sec")
        st.write(f"• Total Samples: {st.session_state.total}")

        # =========================
        # SAVE CSV
        # =========================
        df = pd.DataFrame(st.session_state.records)

        os.makedirs("results", exist_ok=True)

        if os.path.exists(RESULTS_PATH):
            old_df = pd.read_csv(RESULTS_PATH)
            df = pd.concat([old_df, df], ignore_index=True)

        df.to_csv(RESULTS_PATH, index=False)

        st.success("📁 Results saved")

        # =========================
        # ANALYTICS
        # =========================
        st.subheader("📈 Analytics")

        st.write("Accuracy Distribution:")
        st.bar_chart(df["correct"].value_counts())

        st.write("Time Distribution:")
        st.line_chart(df["time"])

        st.write("Time vs Accuracy:")
        st.scatter_chart(df[["time", "correct"]])

        st.dataframe(df.tail(10))

        # Download button
        st.download_button(
            label="📥 Download Results",
            data=df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )

    st.stop()

# =========================
# CURRENT CAPTCHA
# =========================
file = st.session_state.files[st.session_state.index]
path = os.path.join(CAPTCHA_FOLDER, file)

gt = file.split("_")[1].split(".")[0]

img = Image.open(path)

st.image(img, caption=f"CAPTCHA {st.session_state.index + 1}/{len(st.session_state.files)}")

# Real-time score
st.write(f"📊 Score: {st.session_state.correct}/{st.session_state.total}")

# Timer start
if st.session_state.start_time is None:
    st.session_state.start_time = time.time()

# Input label
label = "Enter CAPTCHA"
if show_answer:
    label += f" (Correct: {gt})"

user_input = st.text_input(label)

# =========================
# SUBMIT BUTTON
# =========================
if st.button("Submit"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter CAPTCHA")
        st.stop()

    end_time = time.time()
    elapsed = end_time - st.session_state.start_time

    timeout = elapsed > MAX_TIME
    is_correct = (user_input.strip().upper() == gt) and not timeout

    st.session_state.records.append({
        "timestamp": datetime.now(),
        "user": st.session_state.user,
        "captcha": gt,
        "user_input": user_input,
        "correct": int(is_correct),
        "time": elapsed,
        "timeout": int(timeout)
    })

    st.session_state.times.append(elapsed)

    if timeout:
        st.warning("⏱️ Time exceeded!")
    elif is_correct:
        st.success("✅ Correct")
        st.session_state.correct += 1
    else:
        st.error(f"❌ Wrong (Correct: {gt})")

    st.session_state.total += 1
    st.session_state.index += 1
    st.session_state.start_time = None

    st.rerun()

# =========================
# SKIP BUTTON
# =========================
if st.button("Skip"):

    st.session_state.records.append({
        "timestamp": datetime.now(),
        "user": st.session_state.user,
        "captcha": gt,
        "user_input": "SKIPPED",
        "correct": 0,
        "time": None,
        "timeout": 0
    })

    st.session_state.total += 1
    st.session_state.index += 1
    st.session_state.start_time = None

    st.rerun()

# =========================
# PROGRESS BAR
# =========================
progress = st.session_state.index / len(st.session_state.files)
st.progress(progress)