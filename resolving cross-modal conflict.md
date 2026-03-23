
# Prompting Strategy

- **Incongruity-Aware Prompting**: Take advantage of auto-regressive nature of LLM, force it to do a strict CoT with detail steps to identify type of conflict, for example:
	- 1. Do uni-modal analyzation
	- 2. Explicit Incongruity Acknowledgment
	- 3. Apply psychological rule, e.g. determine if it is sarcasm, deceit, masking, ...
	- 4. Generate soft-label
- **Theory-of-mind**: Treat the conflict not as a error, but as a deliberate communication act, use CoT to prompt the LLM to analyze the speaker's hidden mental state (reason, want, intention, ...) before generating the final label, **must have** contextual information about the conversation.
- **Multi-Agent Debate**: Same as ToM, but use multi-agent to reduce the load on 1 LLM, can have each debater apply ToM on each modality
- **Soft-Labeling**: Instead of forcing the model to collapse a complex conflict into a single basic emotion, let it output a probability distribution.

# Non-prompting Strategy
- **Cross-modal Attention Mechanisms**
- **Inference-Time Attention Reallocation**