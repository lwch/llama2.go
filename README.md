# llama2.go

Port of Facebook's LLaMA model in pure go and use little memory.

## memory usage

| Model | Precision | Memory | Memory(Cached Params) |
| ----- | --------- | ------ | --------------------- |
| 7B  | bf16 | 600M+ | 25G+ |
| 13B | bf16 | 1G+ | 43G+ |
| 70B | bf16 | 3G+ | untest |

## usage

1. download model from [huggingface](https://huggingface.co/lwch/llama2.go)
2. build this project
    ```shell
    go build
    ```
3. text completion
    ```shell
    cat << EOF | ./llama2 text-completion -m 7B.model [--cache]
    Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>
    EOF
    2023/10/16 13:44:47 [INFO]model loaded
    2023/10/16 13:44:47 [INFO]model info:
    2023/10/16 13:44:47 [INFO]  + embedding dim: 4096
    2023/10/16 13:44:47 [INFO]  + layers: 32
    2023/10/16 13:44:47 [INFO]  + heads: 32
    2023/10/16 13:44:47 [INFO]  + kv_heads: 32
    2023/10/16 13:44:47 [INFO]  + q_embedding each head: 128
    2023/10/16 13:44:47 [INFO]  + kv_embedding each head: 128
    2023/10/16 13:44:47 [INFO]  + norm_eps: 1.000000e-05
    2023/10/16 13:44:47 [INFO]warm up model...
    2023/10/16 13:45:00 [INFO]warm up done
    cost: 15.373085s, prompt: [<s>], inference: [ The]
    cost: 15.510916371s, prompt: [Tra], inference: [de]
    cost: 16.684040418s, prompt: [ns], inference: [fer]
    cost: 15.359805539s, prompt: [late], inference: [.]
    cost: 15.175803776s, prompt: [ Eng], inference: [l]
    cost: 15.53356004s, prompt: [lish], inference: [ to]
    cost: 15.81259559s, prompt: [ to], inference: [ Chinese]
    cost: 15.773628902s, prompt: [ Fre], inference: [ch]
    cost: 15.074264719s, prompt: [nc], inference: [sh]
    cost: 16.118718266s, prompt: [h], inference: [,]
    cost: 16.625596815s, prompt: [:], inference: [
    ]
    cost: 15.313741992s, prompt: [
    ], inference: [Trans]
    cost: 14.911782235s, prompt: [
    ], inference: [
    ]
    cost: 15.618974561s, prompt: [se], inference: [ven]
    cost: 16.623896267s, prompt: [a], inference: [ to]
    cost: 15.726856668s, prompt: [ ott], inference: [oman]
    cost: 15.556770908s, prompt: [er], inference: [
    ]
    cost: 15.83002783s, prompt: [ =>], inference: [ se]
    cost: 16.109959208s, prompt: [ lo], inference: [b]
    cost: 16.127250925s, prompt: [ut], inference: [
    ]
    cost: 15.24370557s, prompt: [re], inference: [
    ]
    cost: 16.382684091s, prompt: [ de], inference: [ mer]
    cost: 16.664592192s, prompt: [ mer], inference: [
    ]
    cost: 16.717715045s, prompt: [
    ], inference: [
    ]
    cost: 16.034261912s, prompt: [pe], inference: [ace]
    cost: 16.521538864s, prompt: [pper], inference: [ =>]
    cost: 16.802833067s, prompt: [min], inference: [st]
    cost: 16.591869046s, prompt: [t], inference: [ =>]
    cost: 16.906193174s, prompt: [ =>], inference: [ po]
    cost: 16.869097755s, prompt: [ ment], inference: [he]
    cost: 17.293059622s, prompt: [he], inference: [ po]
    cost: 16.354959941s, prompt: [ poi], inference: [il]
    cost: 17.689774477s, prompt: [vr], inference: [ée]
    cost: 17.598683193s, prompt: [ée], inference: [
    ]
    cost: 17.703475828s, prompt: [
    ], inference: [
    ]
    cost: 16.798081786s, prompt: [pl], inference: [um]
    cost: 17.651892141s, prompt: [ush], inference: [ =>]
    cost: 18.19274064s, prompt: [ gir], inference: [lt]
    cost: 16.960221272s, prompt: [af], inference: [ =>]
    cost: 18.512423328s, prompt: [e], inference: [ =>]
    cost: 17.775961491s, prompt: [ =>], inference: [ pl]
    cost: 18.683528258s, prompt: [ gir], inference: [raf]
    cost: 17.250048314s, prompt: [af], inference: [on]
    cost: 19.24387802s, prompt: [e], inference: [ so]
    cost: 18.543145987s, prompt: [ pel], inference: [uche]
    cost: 18.242073801s, prompt: [uche], inference: [
    ]
    cost: 18.668344731s, prompt: [
    ], inference: [
    ]
    cost: 18.583391073s, prompt: [che], inference: [ese]
    cost: 18.360519174s, prompt: [ese], inference: [ c]
    cost: 17.821802217s, prompt: [ =>], inference: [ from]
    cost: 18.866436719s, inference: [age]
    cost: 18.799781628s, inference: [
    ]
    cost: 18.32387751s, inference: [
    ]
    cost: 19.296660746s, inference: [Tra]
    cost: 19.325703181s, inference: [ans]
    cost: 18.32296003s, inference: [late]
    cost: 19.844884938s, inference: [ French]
    cost: 19.77687613s, inference: [ to]
    cost: 19.460023352s, inference: [ English]
    cost: 18.846496273s, inference: [:]
    cost: 20.903992619s, inference: [
    ]
    cost: 19.66799173s, inference: [
    ]
    cost: 18.746945649s, inference: [lait]
    cost: 20.682075679s, inference: [ =>]
    cost: 20.126048849s, inference: [ milk]
    cost: 19.620849292s, inference: [
    ]
    [Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>] ==>  fromage

    Traanslate French to English:

    lait => milk
    ```
4. chat
    ```shell
    ./llama2 chat -m 7B.chat.model [--cache]
    2023/10/17 10:13:50 [INFO]model loaded
    2023/10/17 10:13:50 [INFO]model info:
    2023/10/17 10:13:50 [INFO]  + embedding dim: 4096
    2023/10/17 10:13:50 [INFO]  + layers: 32
    2023/10/17 10:13:50 [INFO]  + heads: 32
    2023/10/17 10:13:50 [INFO]  + kv_heads: 32
    2023/10/17 10:13:50 [INFO]  + q_embedding each head: 128
    2023/10/17 10:13:50 [INFO]  + kv_embedding each head: 128
    2023/10/17 10:13:50 [INFO]  + norm_eps: 1.000000e-06
    2023/10/17 10:13:50 [INFO]warm up model...
    2023/10/17 10:14:10 [INFO]warm up done
    Enter system prompt (optional):
    Enter user prompt: Whats's your name?
    thinking.................
    Hello! My name is LLaMA, I'm a large language model trained by a team of researcher at Meta AI. ð
    Enter user prompt: Where are you from?
    thinking................
    I'm just an AI, I don't have a physical body or a specific location where I "come from."" I was created by a group of researcher at Meta AI and my primary function is to assist and converse with users like you through the internet. I'm excited to be here and help you with any questions or topics you'd like to discuss!
    ```

Note: if you have enough memory you can run with `--cache` param to persist params in memory

## convert model by yourself

[request](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and download llama2 model, the model directory like this.

    .
    ├── llama-2-7b
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model

using command below to convert llama model to llama2 model fot this project.

NOTE: make sure you have enough hard disk space.

```shell
./llama2 convert --output 7B.model ~/llama/llama-2-7b
```

NOTE: support quantization in the feature