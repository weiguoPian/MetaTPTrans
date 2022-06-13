# MetaTPTrans

This repository is based on the implementation of TPTrans.

## Raw data
To run experiments, please first create datasets from raw code snippets of [CodeSearchNet dataset](https://github.com/github/CodeSearchNet).
Download and unzip the raw jsonl data of CSN into the _raw_data_ dir like that
```yaml
├── raw_data     
│   ├── python         
│   │   ├── train    
│   │   │   ├── XXXX.jsonl...
│   │   ├── test    
│   │   ├── valid   
│   ├── ruby          
│   ├── go        
│   ├── javascript        
```
For the subset used for code completion task, please download it [here](https://drive.google.com/file/d/1HmZviSzje-STZaBH8kvt_bLEFPorEcFr/view?usp=sharing) and parse it.

## Preprocess

### AST Parser
We use Tree-Sitter to parse the source code snippets to ASTs. Please put the parser into vendor fold like this.
```yaml
├── vendor        
│   ├── tree-sitter-python  (from https://github.com/tree-sitter/tree-sitter-python)         
│   ├── tree-sitter-javascript  (from https://github.com/tree-sitter/tree-sitter-javascript)     
│   ├── tree-sitter-go  (from https://github.com/tree-sitter/tree-sitter-go)
│   ├── tree-sitter-ruby  (from https://github.com/tree-sitter/tree-sitter-ruby)

```


And then, run script *multi_language_parse.py* for preprocessing data for code summarization task.

And run *multi_language_parse_completion.py* (if applicable) for preprocessing data for code completion task.