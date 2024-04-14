# lang

Remove the conda environment `LLM_LangChain_Unstructured` and create it again

```
conda remove -y -n LLM_LangChain_Unstructured --all &&
conda env create -y -f environment.yml
```

Rename `config_example.ini` file to `config.ini` and fill the blanks

