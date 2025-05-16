`run-med-evidence` is a separately-contained setup to actually run the evaluation (assuming you have the relevant API keys or vllm hosting setup to do so). it has a separate environment defined by `run-med-evidence/requirements.txt`, which can be installed as a python venv.

The outer directory is used to create the dataset and analyze llm outputs, and uses a separate environment defined by the `medenv.yml` file. Figures and analysis were performed using the `inspect_outputs.ipynb` file.

We include the initial subset of 250 questions used to perform initial analysis in the run-med-evidence/datasets folder for consistency. Please note that our separate dataset submission provides the full set of data in a separate (more useful) format with additional metadata that, in our case, was extracted during the analysis process.
