@dataclass
class ActivationContextModelConfig:
    side_model_name: str
    side_model_lora_name: str
    target_model_name: str
    compression_target: int = 16
    input_pooling_factor: int = 4
    input_pooling_window_scale: int = 4 # multiplies the input pooling factor.
    view_pooling_window_scale: int = 4 # multiplies the compression target.
    checkpoint_path: str  # Where to save/load from.


class ActicationContextModel: # This is now the one canonical activation context model. Delete the retrieval-specifc path.
    """
    Accepts input of the form:
    messages = [
        {
            "role": "x",
            "content": {
                {"type": "text", "text": "Here is subcontent1",
                {"type": "activation_context", "messages": [...]},
                {"type": "text", "text": "Here is subcontent2"}, 
                {"type": "activation_context", "messages": [...]},
            }
        },
        ...
    ]
    and produces embedding tensor.
    The recursive AC messages are independently passed into this model to get their values, ran into a standard ffn-based adapter, then passes in as embeddings at their right full position in the input stream.

    It's important to have a very simple (cpu) cache of hash -> final tensor (NOT inner kvs). Just that given the same messages input (whether top-level or recursive), avoid recomputation.

    It's also important to support (mostly through the utils files), schedule and batching that makes parallel calls to forward be very efficient. I don't know whether there should be a different mode for training and rollouts. I also don't know if we should reduce engine occupancy by ~5-10% to give room for this (is vllm literraly hogging things in way that makes any other side cache impossible?)

    Tentative first architecture (give me an html visualization so I know we agree this makes sense).
    - Run the recursive steps, adapt their output, then tokenize with placeholders where these outputs would go.
        - Cache comes into play at recursive steps here or at the adapt step.
    - Run the tokens through the frozen embeddings layer of the base model. Replace the embeddings.
    - Do a self-attention based pooling to get the main major reduction in computation.
    - At the same time, on do a larger pooling pass to create the view tokens. This creates the target number of tokens + a learned view marker. Append them to the stream.
    - You now have <N/P_initial, N/P_target> tokens.
    - Don't forget to add any necessary positional embeddings.
    - Forward through the model as usual and read out the view tokens + some kind of head ffn (different from the recursive adapter; this targets the generative model, not the AC itself).
    - Attach a lora to the base model.
    - I am undecided on whether remove causality is worth it (embedding models don't seem to care either way, but this is slightly different). Let's defer if it introduces too many hacks to make it work. But let's do it if it's just a flag change.
    """
    def __init__(
        self,
        harness: HarnessRuntime,
        ac_model_config: ActivationContextModelConfig,
    ):
        pass # Every other parameters should just be some unconfigurable, standard.


    def encode(
        messages: list[dict]|str, # support simple string input by converting it to a user message.
    ):
        pass


    def encode_batch(
        messages: list[dict]|str, # ...
        for_training: bool, # needed ???
    ):
        pass