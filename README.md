# llama2.go

Port of Facebook's LLaMA model in go

## prepare

[request download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and download llama2 model, the model directory like this.

    .
    ├── llama-2-7b
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model

## usage

1. install [gotorch](https://github.com/lwch/gotorch#安装)
2. build llama2 command
    ```shell
    go build
    ```
3. convert checkpoint to tnn model
    ```shell
    # TODO: support quantize
    ./llama2 convert --model ~/llama --name llama-2-7b --output ./models

    # output directory
    .
    ├── llama2.model
    ├── params.json
    └── tokenizer.model
    ```

4. text completion
    ```shell
    echo "PyTorch is" | ./llama2 text-completion --model ./models

    2023/10/10 17:34:13 [INFO]loading tokenizer from models/tokenizer.model...
    2023/10/10 17:34:13 [INFO]tokenizer model loaded, token size: 32000
    2023/10/10 17:34:13 [INFO]loading params from models/params.json...
    2023/10/10 17:34:13 [INFO]loading model from models/llama2.model...
    2023/10/10 17:34:28 [INFO]model loaded
    PyTorch is a Python library for the creation of a ...
    ```