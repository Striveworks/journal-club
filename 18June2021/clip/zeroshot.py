import os
import pdb
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    ToPILImage,
    Normalize,
)
from torchvision.utils import make_grid

from .model import build_model
from .tokenizer import SimpleTokenizer
from .utils import softmax, directory_chunker


class CLIP:
    """
    A zeroshot model for accepting a user's
    free text input and returning the most relevant
    images within a supplied directory.

    Parameters:
    -----------
    model_path: str
    """

    def __init__(self, model_path: str, vocab_path: str, gpu_available: bool = None):
        # self.gpu_available = gpu_available or torch.cuda.is_available()
        self.gpu_available = False
        self.device = "cpu" if not self.gpu_available else "cuda"
        self.model = torch.jit.load(
            model_path, map_location=self.device).eval()
        if not self.gpu_available:
            self.model = build_model(self.model.state_dict()).float()
            self.input_resolution = self.model.visual.input_resolution
        else:
            self.input_resolution = self.model.input_resolution.item()
        self.model = self._patch_device(self.model)
        if not self.gpu_available:
            self.model = self._patch_cpu_model(self.model)
        self.preprocess = self._transform(self.input_resolution)
        self.tokenizer = SimpleTokenizer(byte_pair_encoding_path=vocab_path)

    def _transform(self, npx: int):
        """
        Transformation pipeline for images.

        Parameters:
        -----------
        npx: int
            Input resolution.

        Returns:
        --------
        torchvision.transforms.Compose
        """
        return Compose(
            [
                Resize(npx, interpolation=Image.BICUBIC),
                CenterCrop(npx),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _patch_device(self, model):
        """
        Patch device names.

        Parameters:
        -----------
        model: torch.jit.model

        Returns:
        --------
        model: torch.jit.model
        """
        device_holder = torch.jit.trace(
            lambda: torch.ones([]).to(torch.device(self.device)), example_inputs=[]
        )
        device_node = [
            n
            for n in device_holder.graph.findAllNodes("prim::Constant")
            if "Device" in repr(n)
        ][-1]

        def patch_device(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(
                        node["value"]
                    ).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)
        return model

    def _patch_cpu_model(self, model):
        """
        Patch dtype to float32 for cpu usage.
        Parameters:
        -----------
        model: torch.jit.model
        Returns:
        --------
        model: torch.jit.model
        """
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()
        return model

    def preprocess_image_directory(
        self, directory: Union[str, None], images: List[str] = None
    ) -> List[torch.tensor]:
        """
        Open images and preprocess using torchvision transforms.

        Parameters:
        -----------
        directory: Union[str, None]
        images: Union[List[str], None]

        Returns:
        --------
        ret: List[torch.tensor]
        """
        if not images:
            images = os.listdir(directory)
        if not isinstance(images, list):
            images = [images]
        if directory:
            filenames = [
                os.path.join(directory, i)
                for i in images
                if i.endswith(".png") or i.endswith(".jpg")
            ]
        else:
            filenames = [i for i in images if i.endswith(
                ".png") or i.endswith(".jpg")]
        processed_images = [
            self.preprocess(Image.open(i).convert("RGB")) for i in tqdm(filenames)
        ]
        return filenames, processed_images

    def encode_images(self, images: List[torch.tensor]) -> np.ndarray:
        """
        Encode images to feature representation space. Assumes
        images are already preprocessed.

        Parameters:
        -----------
        images: List[torch.tensor]

        Returns:
        --------
        image_features: np.ndarray
        """
        image_input = torch.tensor(np.stack(images))
        if self.gpu_available:
            image_input = image_input.cuda()

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        return image_features

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to feature representation space.

        Parameters:
        -----------
        texts: List[str]

        Returns:
        --------
        text_features: torch.tensor[cpu]
        """
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        batch_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]
        text_input = torch.zeros(
            len(batch_tokens), self.model.context_length, dtype=torch.long
        )
        for i, tokens in enumerate(batch_tokens):
            text_input[i, : len(tokens)] = torch.tensor(tokens)
        if self.gpu_available:
            text_input = text_input.cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
        return text_features

    def similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Compute the cosine similarity / probabilities.

        Parameters:
        ----------
        text_features: np.ndarray
        image_features: np.ndarray
        return_probabilities: bool

        Returns:
        --------
        x: np.ndarray
        """
        if return_probabilities:
            """Cosine similarity as logits"""
            logits_per_image = 100.0 * (image_features @ text_features.T)
            image_probabilities = softmax(logits_per_image, axis=-1)
            return image_probabilities
        return image_features @ text_features.T

    def top(
        self,
        text_features: torch.tensor,
        image_features: torch.tensor,
        images: List[torch.tensor],
        topk: int,
    ) -> List[Tuple[float, torch.tensor]]:
        """
        Compute similarity between image and text features

        Parameters:
        ----------
        text_features: torch.tensor
        image_features: torch.tensor
        images: List[torch.tensor]
        topk: int

        Returns:
        --------
        topk_pairs: List[Tuple[float, torch.tensor]]

        """
        similarity = self.similarity(
            text_features, image_features, return_probabilities=False
        )
        flattened_similarity = similarity.flatten()
        maxk = similarity.shape[1]
        take = min(maxk, topk)
        indices = np.argpartition(flattened_similarity, -take)[-take:]
        pairs = list(zip([flattened_similarity[i] for i in indices], indices))
        sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        sorted_probabilities = [sp[0] for sp in sorted_pairs]
        sorted_take_indices = [sp[1] for sp in sorted_pairs]
        topk_pairs = [
            [sorted_probabilities[i], images[j]]
            for i, j in enumerate(sorted_take_indices)
        ]
        return topk_pairs

    def search(self, image_directories: List[str], prompt: str, topk: int):
        """
        Search a bucket of images for the topk images
        related to a prompt.

        Parameters:
        -----------
        image_directories: List[str]
            Paths to image directories.
        prompts: str
        topk: int

        Returns:
        --------
        topk_images: List[torch.tensor]
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
        text_features = self.encode_texts(prompt)

        n_chunks = 0
        bucket_topk_pairs = []

        for directory in image_directories:
            chunk_generator = directory_chunker(directory, 500)
            for chunk in chunk_generator:
                _, processed_images = self.preprocess_image_directory(
                    directory, chunk)
                image_features = self.encode_images(processed_images)
                topk_pairs = self.top(
                    text_features, image_features, processed_images, topk
                )
                """ Manage the bucket topk images and probs """
                bucket_topk_pairs.extend(topk_pairs)
                bucket_topk_pairs = sorted(
                    bucket_topk_pairs, key=lambda x: x[0], reverse=True
                )
                bucket_topk_pairs = bucket_topk_pairs[:topk]
                n_chunks += 1
                print(f"processed {n_chunks} chunk(s)")

        return bucket_topk_pairs

    @classmethod
    def build_default_clip(cls, gpu_available: bool = False):
        root = Path().cwd() / Path(__file__).resolve().parent.parent
        clip_model_path = root / "models/vit.pt"
        vocab_path = root / "vocab/bpe_simple_vocab_16e6.txt.gz"
        return cls(
            model_path=str(clip_model_path),
            vocab_path=str(vocab_path),
            gpu_available=gpu_available
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    """
    Sample Call:
        python3 -m clip.zeroshot --prompt 'show me cats' --topk 3 --out-path ~/clip.jpg
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()

    clip = CLIP.build_default_clip()

    directories = ["/home/ttheisen/clippy/static/images"]

    pairs = clip.search(directories, [str(args.prompt)], int(args.topk))

    topk_images = [i[1] for i in pairs]
    grid = make_grid(torch.stack((topk_images)))

    def tensor_to_pil(t):
        return ToPILImage()(t).convert("RGB")

    grid = tensor_to_pil(grid)
    grid.save(args.out_path)
