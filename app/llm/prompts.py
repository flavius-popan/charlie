"""Message formatting utilities for Graphiti LLM bridge."""

from graphiti_core.prompts.models import Message


def format_messages(messages: list[Message], tokenizer) -> str:
    """
    Convert Graphiti messages to chat prompt string.

    Strategy:
    1. Try native chat template if available (preferred)
    2. Fallback to simple System:/User:/Assistant: format

    Args:
        messages: List of Graphiti Message objects
        tokenizer: MLX tokenizer (may have apply_chat_template)

    Returns:
        Formatted prompt string ready for generation
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        # Use native chat template from model
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple text format
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
        prompt += "Assistant:"
        return prompt
