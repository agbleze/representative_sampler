import torch
import clip
from PIL import Image
import numpy as np
import sklearn.preprocessing as skp
from representative_sampler.core.domain.strategies.base_embedder import Embedder






class ClipEmbedder(Embedder) -> EmbeddingResult:
    name = "clip"
    def __init__(self, img_list, model_type):
        self.img_list = img_list
        self.model_type = model_type
        ClipEmbedder.name = f"clip-{model_type}"
        
    def embed(self):
        self.normalized_embedding = self.get_embeddings(self.img_list, self.model_type)
        return self.normalized_embedding
    
    def get_embeddings(self, img_list, model_type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_type, device=device)

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
        return normalized_embedding
    
    



