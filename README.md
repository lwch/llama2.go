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