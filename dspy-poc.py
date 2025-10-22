import dspy

lm = dspy.LM(
    "openai/qwen/qwen3-4b-2507",
    api_base="http://127.0.0.1:8000/v1",
    api_key="LOCALAF",
)
dspy.configure(lm=lm)

print(lm("Say this is a test!", temperature=0.7))
