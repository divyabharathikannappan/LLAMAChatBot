import numpy as np
import pandas as pd
import plotly.express as px
import os
import webbrowser
from datetime import datetime
from sklearn.decomposition import PCA
from langchain_community.vectorstores import Chroma
import re  # Import regex for title extraction
from pathlib import Path

def extract_title(doc_content: str) -> str:
    """Extracts the Grant Name from the document content."""
    match = re.search(r"Grant Name:\s*(.*)", doc_content)
    return match.group(1).strip() if match else "Unknown Title"

def visualize(doc_db: Chroma, output_dir: str = ".") -> None:
    """Create interactive visualization of embeddings using Plotly."""
    try:
        collection = doc_db._collection
        if not collection:
            print("[ERROR] No collection available in document database")
            return
        
        result = collection.get(
            include=['embeddings', 'documents', 'metadatas']
        )
        
        if not result or 'embeddings' not in result or not result['embeddings']:
            print("[ERROR] No embeddings found or embeddings list is empty in collection")
            return
        
        embeddings = np.array(result['embeddings'], dtype=np.float64)
        if embeddings.ndim == 1: # Handle case where only one embedding is present
             print("[WARNING] Only one embedding found. PCA requires at least 2 samples. Skipping visualization.")
             return
        if embeddings.shape[0] < 2:
             print("[WARNING] Less than 2 embeddings found. PCA requires at least 2 samples. Skipping visualization.")
             return

        # Ensure documents are present and match the number of embeddings
        if 'documents' not in result or len(result['documents']) != len(embeddings):
             print("[ERROR] Mismatch between number of embeddings and documents, or documents are missing.")
             return

        titles = [extract_title(doc) for doc in result['documents']]
        
        # Ensure PCA n_components is not more than the number of samples or features
        n_components = min(2, embeddings.shape[0], embeddings.shape[1])
        if n_components < 2:
             print(f"[WARNING] Cannot perform PCA with n_components={n_components}. Skipping visualization.")
             return
             
        pca = PCA(n_components=n_components)
        
        try:
            reduced_embeddings = pca.fit_transform(embeddings)
        except ValueError as ve:
            print(f"[ERROR] PCA failed: {ve}. Skipping visualization.")
            return

        df_data = {
            'Title': titles,
        }
        # Add PCA components dynamically based on n_components
        for i in range(n_components):
            df_data[f'PC{i+1}'] = reduced_embeddings[:, i]

        df = pd.DataFrame(df_data)

        # Adjust plot based on n_components
        # DataFrame 'df' will be passed as the first argument to px.scatter
        plot_params = {
            'hover_data': ['Title'],
            'title': 'Grant Embeddings PCA Visualization'
        }
        if n_components == 1:
             # If only 1 component, maybe plot against index or a constant y?
             # For simplicity, we'll just print a message and skip plotting 1D for now.
             print("[INFO] PCA resulted in only 1 dimension. Skipping 2D scatter plot.")
             return
        elif n_components >= 2:
             plot_params['x'] = 'PC1'
             plot_params['y'] = 'PC2'

        # Pass DataFrame 'df' as the first argument, unpack the rest
        fig = px.scatter(df, **plot_params)

        fig.update_traces(
            hovertemplate="<br>".join([
                "Title: %{customdata[0]}",
                # Add other hover data if needed, e.g., PC values
                "PC1: %{x}",
                "PC2: %{y}"
            ])
        )
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        viz_filename = f"embeddings_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        viz_path = os.path.abspath(output_path / viz_filename)
        
        try:
            fig.write_html(viz_path)
            print(f"[SUCCESS] Visualization saved to {viz_path}")
            webbrowser.open(f'file://{viz_path}')
        except Exception as write_e:
             print(f"[ERROR] Failed to write or open visualization HTML: {write_e}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during visualization: {e}")
        # Optionally re-raise if you want the main script to handle it further
        # raise
