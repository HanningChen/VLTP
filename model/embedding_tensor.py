import torch
import cv2
import numpy as np


class embedding_tensor(torch.nn.Module):
    def __init__(self, 
                 image_embeddings, 
                 image_pe, 
                 sparse_embeddings,
                 dense_embeddings,
                 ):
        super().__init__()
        self.image_embeddings = torch.nn.Parameter(image_embeddings,requires_grad=False)
        self.image_pe = torch.nn.Parameter(image_pe,requires_grad=False)
        self.sparse_embeddings = torch.nn.Parameter(sparse_embeddings,requires_grad=False)
        self.dense_embeddings = torch.nn.Parameter(dense_embeddings,requires_grad=False)

    def forward(self):
        return self.image_embeddings, self.image_pe, self.sparse_embeddings, self.dense_embeddings