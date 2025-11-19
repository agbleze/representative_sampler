import torch
import clip
from PIL import Image
import numpy as np
import sklearn.preprocessing as skp
from .base_embedder import Embedder
from representative_sampler.core.entities import EmbeddingResult


_SENTINEL = object()

class ClipEmbedder(Embedder):
    embedder_name = "clip"
    status = "stable"
    def __init__(self,
                 model_type
                 ):
        self.model_type = model_type
        
    def embed(self)-> EmbeddingResult:
        self.normalized_embedding = self.get_embeddings(self.img_list, self.model_type)
        
        embedding_result = EmbeddingResult(embedder_name=ClipEmbedder.name,
                                            embedding_name=self.img_list,
                                            embedding=self.normalized_embedding
                                            )
        return embedding_result
    
    
    def get_embeddings(self, img_list, model_type=_SENTINEL)-> EmbeddingResult:
        if model_type is _SENTINEL:
            model_type = self.model_type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(name=model_type, device=device)

        embeddings = []
        for img_path in img_list:
            img = Image.open(img_path)
            image = preprocess(img=img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                embeddings.append(image_features)
                
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                            axis=1
                                            )
        
        embedding_result = EmbeddingResult(embedder_name=self.embedder_name,
                                            embedding_name=img_list,
                                            embedding=normalized_embedding
                                            )
        return embedding_result