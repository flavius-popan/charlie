import dspy
from pydantic import BaseModel
from dspy_outlines.base_lm import PassthroughLM

def test_passthrough_lm_basic_call():
    """Test that PassthroughLM can handle basic DSPy calls."""
    lm = PassthroughLM(
        model="openai/qwen/qwen3-4b-2507",
        api_base="http://127.0.0.1:8000/v1",
        api_key="LOCALAF"
    )
    dspy.configure(lm=lm)

    class SimpleOutput(BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: SimpleOutput = dspy.OutputField()

    predictor = dspy.Predict(SimpleSignature)
    result = predictor(question="What is 2+2?")

    assert hasattr(result, 'answer')
    assert isinstance(result.answer, SimpleOutput)
