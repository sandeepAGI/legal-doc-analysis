from sentence_transformers import SentenceTransformer
import os

# Base model directory
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"

print("🔄 Downloading all embedding models...")
print("=" * 50)

# Download BGE models
print("📥 Downloading BGE models...")
print("🔄 BGE Small...")
model = SentenceTransformer("BAAI/bge-small-en")
model.save(os.path.join(MODEL_ROOT, "bge-small-en"))
print("✅ BGE Small saved")

print("🔄 BGE Base...")
model = SentenceTransformer("BAAI/bge-base-en")
model.save(os.path.join(MODEL_ROOT, "bge-base-en"))
print("✅ BGE Base saved")

# Download Arctic Embed model
print("📥 Downloading Arctic Embed model...")
print("🔄 Arctic Embed 33m...")
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")
model.save(os.path.join(MODEL_ROOT, "arctic-embed-33m"))
print("✅ Arctic Embed saved")

# Download all-MiniLM model
print("📥 Downloading all-MiniLM model...")
print("🔄 all-MiniLM-L6-v2...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save(os.path.join(MODEL_ROOT, "all-minilm-l6-v2"))
print("✅ all-MiniLM-L6-v2 saved")

print("=" * 50)
print("🎉 All embedding models downloaded successfully!")
print("📁 Models saved to:", MODEL_ROOT)
print("\nAvailable models:")
print("  - bge-small-en (BGE Small)")
print("  - bge-base-en (BGE Base)") 
print("  - arctic-embed-33m (Arctic Embed 33m)")
print("  - all-minilm-l6-v2 (all-MiniLM-L6-v2)")