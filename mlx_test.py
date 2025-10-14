from pydantic import BaseModel
from typing import List
import outlines
import mlx_lm


class Character(BaseModel):
    name: str
    age: int
    skills: List[str]


model = outlines.from_mlxlm(*mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit"))

result = model("Create a character.", output_type=Character)
print(
    result
)  # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(
    Character.model_validate_json(result)
)  # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy'
