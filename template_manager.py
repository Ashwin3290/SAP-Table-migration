import os
import json
import re
import logging
import hashlib
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Template:
    """Represents a generalized code or plan template"""
    
    def __init__(self, template_type, template_content, original_query="", metadata=None):
        self.template_type = template_type  # 'plan', 'code', etc.
        self.template_content = template_content
        self.original_query = original_query
        self.generalized_query = self._generalize_query(original_query)
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.id = self._generate_id()
        
    def _generate_id(self):
        """Generate a unique ID for the template"""
        content_hash = hashlib.md5(self.template_content.encode()).hexdigest()
        return f"{self.template_type}_{content_hash[:10]}"
        
    def _generalize_query(self, query):
        """Generalize a query by replacing specific terms with placeholders"""
        # Replace specific column names with placeholders
        # This is a simplified version - in practice, you'd need more sophisticated NLP
        generalized = query
        
        # Replace specific table names
        table_pattern = r'\b[A-Z][A-Z0-9_]{2,}\b'  # Simple pattern for SAP table names
        tables = re.findall(table_pattern, query)
        for i, table in enumerate(tables):
            generalized = generalized.replace(table, f"TABLE_{i+1}")
            
        # Replace specific field names
        field_pattern = r'\b[A-Z][A-Z0-9_]+\b'  # Simple pattern for SAP field names
        fields = re.findall(field_pattern, generalized)
        for i, field in enumerate(fields):
            if field.startswith("TABLE_"):
                continue
            generalized = generalized.replace(field, f"FIELD_{i+1}")
            
        return generalized
        
    def to_dict(self):
        """Convert template to dictionary"""
        return {
            "id": self.id,
            "template_type": self.template_type,
            "template_content": self.template_content,
            "original_query": self.original_query,
            "generalized_query": self.generalized_query,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create template from dictionary"""
        template = cls(
            template_type=data.get("template_type"),
            template_content=data.get("template_content"),
            original_query=data.get("original_query"),
            metadata=data.get("metadata")
        )
        template.generalized_query = data.get("generalized_query", "")
        template.created_at = data.get("created_at")
        template.id = data.get("id")
        
        return template

class TemplateManager:
    """Manages code and plan templates"""
    
    def __init__(self, storage_path="templates"):
        self.storage_path = storage_path
        self.model = None  # Lazy load the embedding model
        
        try:
            os.makedirs(storage_path, exist_ok=True)
            os.makedirs(f"{storage_path}/plan", exist_ok=True)
            os.makedirs(f"{storage_path}/code", exist_ok=True)
            os.makedirs(f"{storage_path}/embeddings", exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create template storage directories: {e}")
            raise Exception(f"Failed to create template storage: {e}")
            
    def _load_embedding_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            try:
                # Using a smaller model as requested
                self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise Exception(f"Failed to load embedding model: {e}")
                
        return self.model
        
    def create_template(self, template_type, template_content, original_query="", metadata=None):
        """Create and save a new template"""
        try:
            template = Template(
                template_type=template_type,
                template_content=template_content,
                original_query=original_query,
                metadata=metadata
            )
            
            # Save template
            type_dir = f"{self.storage_path}/{template_type}"
            
            with open(f"{type_dir}/{template.id}.json", 'w') as f:
                json.dump(template.to_dict(), f, indent=2)
                
            # Generate and save embedding
            if original_query:
                self._save_embedding(template.id, template_type, original_query)
                
            return template
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return None
            
    def _save_embedding(self, template_id, template_type, text):
        """Generate and save embedding for a template"""
        try:
            model = self._load_embedding_model()
            embedding = model.encode(text)
            
            np.save(f"{self.storage_path}/embeddings/{template_id}.npy", embedding)
            
            # Update embedding index
            index_file = f"{self.storage_path}/embeddings/index.json"
            
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {"templates": []}
                
            index["templates"].append({
                "id": template_id,
                "type": template_type,
                "text": text
            })
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving embedding: {e}")
            return False
            
    def find_similar_templates(self, query, template_type=None, top_k=3):
        """Find templates similar to the query"""
        try:
            model = self._load_embedding_model()
            query_embedding = model.encode(query)
            
            # Load embedding index
            index_file = f"{self.storage_path}/embeddings/index.json"
            
            if not os.path.exists(index_file):
                return []
                
            with open(index_file, 'r') as f:
                index = json.load(f)
                
            # Filter by template type if specified
            if template_type:
                templates = [t for t in index["templates"] if t["type"] == template_type]
            else:
                templates = index["templates"]
                
            if not templates:
                return []
                
            # Calculate similarities
            similarities = []
            
            for template in templates:
                try:
                    template_embedding = np.load(f"{self.storage_path}/embeddings/{template['id']}.npy")
                    similarity = self._cosine_similarity(query_embedding, template_embedding)
                    similarities.append((template, similarity))
                except Exception as e:
                    logger.warning(f"Error calculating similarity for template {template['id']}: {e}")
                    
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_templates = similarities[:top_k]
            
            # Load full templates
            result = []
            for template_info, similarity in top_templates:
                template_file = f"{self.storage_path}/{template_info['type']}/{template_info['id']}.json"
                
                if os.path.exists(template_file):
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        template = Template.from_dict(template_data)
                        
                    result.append({
                        "template": template,
                        "similarity": float(similarity)
                    })
                    
            return result
        except Exception as e:
            logger.error(f"Error finding similar templates: {e}")
            return []
            
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
    def get_template(self, template_id, template_type):
        """Get a template by ID and type"""
        try:
            template_file = f"{self.storage_path}/{template_type}/{template_id}.json"
            
            if not os.path.exists(template_file):
                return None
                
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                
            return Template.from_dict(template_data)
        except Exception as e:
            logger.error(f"Error getting template: {e}")
            return None
            
    def list_templates(self, template_type=None):
        """List all templates, optionally filtered by type"""
        try:
            result = []
            
            if template_type:
                type_dirs = [template_type]
            else:
                type_dirs = os.listdir(self.storage_path)
                type_dirs = [d for d in type_dirs if os.path.isdir(f"{self.storage_path}/{d}") and d != "embeddings"]
                
            for type_dir in type_dirs:
                dir_path = f"{self.storage_path}/{type_dir}"
                
                if not os.path.isdir(dir_path):
                    continue
                    
                for filename in os.listdir(dir_path):
                    if filename.endswith(".json"):
                        template_path = f"{dir_path}/{filename}"
                        
                        with open(template_path, 'r') as f:
                            template_data = json.load(f)
                            
                        result.append(Template.from_dict(template_data))
                        
            return result
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
            
def generalize_code(code_content):
    """
    Replace specific table and field names in code with generic placeholders
    """
    generalized = code_content
    
    # Replace specific table names
    table_pattern = r'\b[A-Z][A-Z0-9_]{2,}\b'  # Simple pattern for SAP table names
    tables = set(re.findall(table_pattern, code_content))
    for i, table in enumerate(tables):
        generalized = re.sub(r'\b' + table + r'\b', f"TABLE_{i+1}", generalized)
        
    # Replace specific field names
    field_pattern = r'\b[A-Z][A-Z0-9_]+\b'  # Simple pattern for SAP field names
    fields = set(re.findall(field_pattern, generalized))
    for i, field in enumerate(fields):
        if field.startswith("TABLE_"):
            continue
        generalized = re.sub(r'\b' + field + r'\b', f"FIELD_{i+1}", generalized)
        
    return generalized
