# Implementation boundary notes

- This pass implements only base-model chat with lora_name=None.
- A non-null LoRA name should raise until registration and routing exist.
- The synchronous API holds the model lifecycle lock through the blocking generation call.
- Level-1 vLLM sleep is treated as parking, not destruction.
- True overlap between chats belongs in a later batched or async engine.
