"""Custom DSPy LM that demonstrates interception (passthrough for now)."""

import dspy
import logging

logger = logging.getLogger(__name__)

class PassthroughLM(dspy.LM):
    """
    Custom LM that passes through to base DSPy.LM.

    This proves we can intercept calls before routing to Outlines.
    """

    def __init__(self, model: str, api_base: str, api_key: str, **kwargs):
        """Initialize by delegating to parent DSPy LM."""
        super().__init__(model, api_base=api_base, api_key=api_key, **kwargs)
        logger.info(f"PassthroughLM initialized: {model}")

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Intercept generation call.

        For now, just log and pass through to parent.
        Future: extract schema, route to Outlines.
        """
        logger.info(f"PassthroughLM.__call__ intercepted")
        logger.debug(f"  prompt: {prompt[:100] if prompt else 'None'}...")
        logger.debug(f"  messages: {len(messages) if messages else 0}")

        # Pass through to parent DSPy.LM
        return super().__call__(prompt=prompt, messages=messages, **kwargs)
