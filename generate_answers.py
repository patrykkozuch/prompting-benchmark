import json
from ollama import generate
from tqdm import tqdm

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

models = [
    {"name": "qwen3:14b", "thinking": True},
    # {"name": "qwen3:1.7b", "thinking": True},
    # {"name": "qwen3:14b", "thinking": False},
    # {"name": "qwen3:14b", "thinking": True}
]

prompt_types = ["zero", "few", "cot"]

combinations = []
for task in data:
    for model in models:
        for pt in prompt_types:
            if model["thinking"] and pt == "cot":
                continue
            combinations.append({"task": task, "model": model, "pt": pt, "thinking": model["thinking"]})

answers = []
for combo in tqdm(combinations):
    task = combo["task"]
    model = combo["model"]
    pt = combo["pt"]
    think = combo["thinking"]

    # Build prompt
    if pt == "zero":
        prompt = f"=== TASK ===\nInput: {task['task']}\nOutput:"
    elif pt == "few":
        exs = "\n\n".join([f"Input: {e[0]}\nOutput: {e[1]}" for e in task["examples"]])
        prompt = f"=== EXAMPLES ===\n{exs}\n\n=== TASK ===\nInput: {task['task']}\nOutput:"
    elif pt == "cot":
        exs = "\n\n".join([f"Input: {e[0]}\nOutput: {e[1]}" for e in task["examples"]])
        prompt = f"=== EXAMPLES ===\n{exs}\n\n=== TASK ===\nInput: {task['task']}\nOutput: Let's think step by step."

    # Send to model
    try:
        output = generate(model=model["name"], system=task["system_prompt"], prompt=prompt, stream=False, options={"num_ctx": 512}, think=think).response
    except Exception as e:
        output = f"Error generating response: {str(e)}"

    answers.append({
        "task_id": task['id'],
        "task_name": task['name'],
        "model": model["name"],
        "thinking": model["thinking"],
        "prompt_type": pt,
        "prompt": prompt,
        "output": output
    })

# Save to file
with open('answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print("Answers generated and saved to answers.json")
