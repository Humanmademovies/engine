from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import logging
from PIL import Image
from typing import List, Optional, Tuple, Iterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fallback_attention():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    logger.info("⚙️ Flash Attention désactivée (fallback).")


class Engine:
    """Moteur universel d'inférence multimodal basé sur MedBot."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Chargement du processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Chargement du modèle (multimodal)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.startswith("cuda") else {"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"[Init] Modèle multimodal chargé : {model_id}")

        # Messages multimodaux
        self.messages: List[dict] = []

    def set_system_prompt(self, system_prompt: str) -> None:
        """Initialise le prompt système."""
        self.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        history: Optional[List[Tuple[str, str]]] = None,
        images: Optional[List] = None,  # Images PIL, chemin ou autre
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
    ) -> str | Iterator[str]:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        if history:
            for u, a in history:
                messages.append({"role": "user", "content": [{"type": "text", "text": u}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})

        content = [{"type": "text", "text": user_prompt}]
        if images is not None:
            for img in images:
                if isinstance(img, str):
                    from PIL import Image
                    img = Image.open(img)
                content.append({"type": "image", "image": img})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)

        if stream:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            import threading
            t = threading.Thread(
                target=self.model.generate,
                kwargs=dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                ),
            )
            t.start()

            for chunk in streamer:
                yield chunk

            t.join()
            return

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )[0][prompt_len:]

        reply = self.processor.decode(ids, skip_special_tokens=True).strip()
        return reply


