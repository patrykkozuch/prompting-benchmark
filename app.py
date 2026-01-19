import streamlit as st
import json

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)


# Load pre-generated answers
with open('answers.json', 'r') as f:
    answers = json.load(f)

answers.sort(key=lambda x: (x['task_id'], x['model'], x['prompt_type']))

models = [
    {"name": "qwen3:1.7b", "thinking": False},
    {"name": "qwen3:1.7b", "thinking": True},
    {"name": "qwen3:14b", "thinking": False},
    {"name": "qwen3:14b", "thinking": True}
]

prompt_types = ["zero", "few", "cot"]

combinations = []
for task in data:
    for model in models:
        for pt in prompt_types:
            if model["thinking"] and pt == "cot":
                continue
            combinations.append({"task": task, "model": model, "pt": pt})

combinations.sort(key=lambda x: (x['task']['id'], x['model']['name'], x['pt']))

if "current" not in st.session_state:
    st.session_state.current = 0
if "scores" not in st.session_state:
    st.session_state.scores = []

st.title("LLM Scoring Test Suite")

if st.session_state.current >= len(combinations):
    st.write("Test suite completed!")
    st.download_button("Download Scores as JSON", json.dumps(st.session_state.scores, indent=2), "scores.json")
else:
    combo = combinations[st.session_state.current]
    task = combo["task"]
    model = combo["model"]
    pt = combo["pt"]
    ans = answers[st.session_state.current]

    st.write(f"Task {task['id']}: {task['name']}")
    st.write("**Task:**", task['task'])
    st.write("**Example Answer:**", task.get('example_answer', 'N/A'))
    st.markdown(ans["output"])

    # Parse scoring instruction
    criteria = task['scoring_instruction']

    checked = []
    st.write("**Scoring Criteria:**")
    for i, crit in enumerate(criteria):
        checked.append(st.checkbox(crit, key=f"check_{st.session_state.current}_{i}"))

    invalid = st.checkbox("Mark as Invalid Output (garbage/empty)", key=f"invalid_{st.session_state.current}")

    if invalid:
        score = 0
    else:
        score = sum(checked) / len(criteria) * 100

    if st.button("Submit Score and Next"):
        st.session_state.scores.append({
            "task_id": task['id'],
            "task_name": task['name'],
            "model": model["name"],
            "thinking": model["thinking"],
            "prompt_type": pt,
            "prompt": ans["prompt"],
            "output": ans["output"],
            "score": score,
            "invalid": invalid
        })
        st.session_state.current += 1
        st.rerun()
