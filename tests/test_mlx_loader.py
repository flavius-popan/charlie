from dspy_outlines.mlx_loader import load_mlx_model, create_outlines_model
from pydantic import BaseModel

def test_load_mlx_model():
    """Test loading Qwen3-4B model via MLX."""
    model, tokenizer = load_mlx_model()

    assert model is not None
    assert tokenizer is not None
    assert hasattr(tokenizer, 'apply_chat_template')

def test_create_outlines_model():
    """Test creating Outlines wrapper."""
    outlines_model, tokenizer = create_outlines_model()

    # Test basic structured generation
    class TestOutput(BaseModel):
        result: int

    response = outlines_model(
        "What is 2+2? Answer with just the number.",
        output_type=TestOutput,
        max_tokens=10
    )

    # Response should be JSON string
    assert isinstance(response, str)
    parsed = TestOutput.model_validate_json(response)
    assert parsed.result == 4
