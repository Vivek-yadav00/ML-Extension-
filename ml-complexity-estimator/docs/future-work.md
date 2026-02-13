# Future Work

## Planned Improvements

1.  **Repo-wide Analysis**: Instead of single files, analyze entire repositories by fetching the file tree via GitHub API.
2.  **Advanced AST Analysis**: Track variable data flow to estimate input shapes (e.g., `(batch, 3, 224, 224)`).
3.  **HuggingFace Integration**: Detect pre-trained model names (e.g., `bert-base-uncased`) and fetch their specific parameter counts.
4.  **Visual Graph**: Generate a visual computation graph for Neural Networks.
5.  **Caching**: Cache analysis results for popular repositories to reduce server load.
