import torch

class StatefulGPT2Wrapper(torch.nn.Module):
    """
    Wraps GPT2 to handle state (KV Cache) explicitly for IREE export.
    Receives current input_ids and past_key_values.
    Returns logits and new_past_key_values.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None):
        # HF GPT2 accepts past_key_values as a tuple of tuples.
        # But for IREE/MLIR export, we usually flatten this list or keep it as a list of tensors.
        # Here we assume the simplifed case where we let torch.export handle the flattening if possible,
        # or we might need to manually flatten/unflatten if the compile fails.
        
        # input_ids: [1, 1] (single token step)
        outputs = self.model(
            input_ids, 
            past_key_values=past_key_values,
            use_cache=True
        )
        return outputs.logits, outputs.past_key_values
