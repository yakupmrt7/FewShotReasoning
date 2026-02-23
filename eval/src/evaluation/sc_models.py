import random

from lmdeploy import GenerationConfig
from lmdeploy.vl import load_image
from PIL import Image

from src.evaluation.lm_prediction_call import LMDeployLM


class LMDeployLMBatch(LMDeployLM):
    """LMDeployLM with an improved generate_n().

    The parent class mutates self.generation_config in the loop and reloads
    the image on every iteration (via the full input_message rebuild). This
    subclass fixes both issues:

      1. Image is loaded ONCE before the loop.
      2. A fresh GenerationConfig is created per iteration (no mutation of
         shared state), with a unique random_seed for output diversity.

    Note: True concurrent batching (sending all N copies in one pipeline call)
    is NOT possible with lmdeploy's VL pipeline. The VL image encoder runs in
    a ThreadPoolExecutor; submitting N concurrent image-encode tasks while
    TurboMind is capturing CUDA graphs triggers:
        RuntimeError: CUDA error: operation not permitted when stream is capturing
    The sequential loop is the correct approach for lmdeploy VL models.
    """

    def generate_n(self, prompt, image_files, n_samples, **kwargs):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(image_files, list):
            image_files = [image_files]

        # Load image(s) ONCE before the loop
        single_messages = []
        for text_message, image_file in zip(prompt, image_files):
            if not isinstance(image_file, Image.Image):
                img = load_image(str(image_file))
            else:
                img = image_file
            single_messages.append((text_message, img))

        # Sequential loop — safe for lmdeploy VL pipeline (no CUDA stream conflict)
        n_responses = []
        for _ in range(n_samples):
            gen_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                random_seed=random.randint(0, 1_000_000),
            )
            response = self.model(single_messages, gen_config=gen_config)
            n_responses.extend(response)

        return [res.text for res in n_responses]
