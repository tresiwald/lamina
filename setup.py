from setuptools import setup, find_packages

setup(
    name="lamina",
    version="0.1.0",
    description=(
        "Modular library for extracting and examining the internal "
        "representations of neural language models — hidden states, "
        "attentions, logits, and logit-lens projections — without "
        "modifying model code or slowing down GPU inference."
    ),
    long_description=open("README.md", encoding="utf-8").read()
        if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "examples*", "notebooks*",
                                     "internals_extraction*"]),
    python_requires=">=3.8",
    # Core has no ML-framework dependency — numpy only
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        # HuggingFace Transformers extractor
        "hf": [
            "transformers>=4.30",
        ],
        # vLLM extractor (future)
        "vllm": [
            "vllm",
        ],
        # Intervention implementations require torch for tensor mutation
        "interventions": [
            "torch",
        ],
        # Additional storage backends
        "backends": [
            "datasets>=2.0",   # HuggingFace datasets for HFDatasetBackend
        ],
        "redis": [
            "redis>=4.0",
        ],
        "mongodb": [
            "pymongo>=4.0",
        ],
        # Install everything
        "all": [
            "transformers>=4.30",
            "datasets>=2.0",
            "torch",
        ],
        # Development / testing
        "dev": [
            "torch",
            "transformers>=4.30",
            "datasets>=2.0",
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
