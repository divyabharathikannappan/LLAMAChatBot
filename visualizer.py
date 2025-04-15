 

import os
import logging
from typing import List, Dict, Any, Optional
from models import Grant
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from langchain.schema import Document
from contextlib import contextmanager
import numpy as np
import plotly.express as px
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import logging.handlers
import configparser
from langchain_community.embeddings import HuggingFaceEmbeddings

def visualize(self, doc_db: Chroma) -> None:
    """Create interactive visualization of embeddings using Plotly."""
    try:
        collection = doc_db._collection
        if not collection:
            raise ValueError("No collection available in document database")
        
        result = collection.get(
            include=['embeddings', 'documents', 'metadatas']
        )
        
        if not result or 'embeddings' not in result:
            raise ValueError("No embeddings found in collection")
        
        embeddings = np.array(result['embeddings'], dtype=np.float64)
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings array")
        
        titles = [self.extract_title(doc) for doc in result['documents']]
        
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        df = pd.DataFrame({
            'PC1': reduced_embeddings[:, 0],
            'PC2': reduced_embeddings[:, 1],
            'Title': titles,
        })
        
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            hover_data=['Title'],
            title='Grant Embeddings PCA Visualization'
        )
        
        fig.update_traces(
            hovertemplate="<br>".join([
                "Title: %{customdata[0]}",
            ])
        )
        
        viz_path = os.path.abspath(self.output_dir / f"embeddings_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        fig.write_html(viz_path)
        
        webbrowser.open(f'file://{viz_path}')
        
    except Exception as e:
        raise
