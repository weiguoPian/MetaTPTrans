# MetaTPTrans

## raw data
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
For the subset used for code completion task, please download it [here](https://drive.google.com/file/d/1HmZviSzje-STZaBH8kvt_bLEFPorEcFr/view?usp=sharing). Or you can directly download the parsed data (see [here](./data/README.MD)).

## Preprocess
Please run script *multi_language_parse.py* for preprocessing data for code summarization task.

And run *multi_language_parse_completion.py* (if applicable) for preprocessing data for code completion task.