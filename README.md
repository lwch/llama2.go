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
    2023/10/19 16:31:50 [INFO]model loaded
    2023/10/19 16:31:50 [INFO]model info:
    2023/10/19 16:31:50 [INFO]  + embedding dim: 4096
    2023/10/19 16:31:50 [INFO]  + layers: 32
    2023/10/19 16:31:50 [INFO]  + heads: 32
    2023/10/19 16:31:50 [INFO]  + kv_heads: 32
    2023/10/19 16:31:50 [INFO]  + q_embedding each head: 128
    2023/10/19 16:31:50 [INFO]  + kv_embedding each head: 128
    2023/10/19 16:31:50 [INFO]  + norm_eps: 1.000000e-05
    2023/10/19 16:31:50 [INFO]warm up model...
    2023/10/19 16:31:57 [INFO]warm up done
    cost: 805.546412ms, prompt: [<s>], inference: [ Tags]
    cost: 894.551053ms, prompt: [Tra], inference: [de]
    cost: 858.299038ms, prompt: [ns], inference: [actions]
    cost: 949.890511ms, prompt: [late], inference: [.]
    cost: 856.886753ms, prompt: [ Eng], inference: [l]
    cost: 840.420843ms, prompt: [lish], inference: [ to]
    cost: 817.256344ms, prompt: [ to], inference: [ Spanish]
    cost: 883.950504ms, prompt: [ Fre], inference: [ch]
    cost: 1.070832166s, prompt: [nc], inference: [ish]
    cost: 1.251877729s, prompt: [h], inference: [
    ]
    cost: 1.246514376s, prompt: [:], inference: [
    ]
    cost: 813.90546ms, prompt: [
    ], inference: [Trans]
    cost: 824.633336ms, prompt: [
    ], inference: [Trans]
    cost: 821.612534ms, prompt: [se], inference: [o]
    cost: 1.234872473s, prompt: [a], inference: [,]
    cost: 997.05027ms, prompt: [ ott], inference: [oman]
    cost: 824.09ms, prompt: [er], inference: [
    ]
    cost: 840.338516ms, prompt: [ =>], inference: [ mer]
    cost: 830.081887ms, prompt: [ lo], inference: [b]
    cost: 830.140757ms, prompt: [ut], inference: [ier]
    cost: 842.806573ms, prompt: [re], inference: [
    ]
    cost: 833.1149ms, prompt: [ de], inference: [ mer]
    cost: 863.183364ms, prompt: [ mer], inference: [
    ]
    cost: 820.171781ms, prompt: [
    ], inference: [How]
    cost: 814.556694ms, prompt: [pe], inference: [ace]
    cost: 853.469381ms, prompt: [pper], inference: [ =>]
    cost: 829.748698ms, prompt: [min], inference: [te]
    cost: 1.224857119s, prompt: [t], inference: [ =>]
    cost: 832.310822ms, prompt: [ =>], inference: [ po]
    cost: 917.125358ms, prompt: [ ment], inference: [he]
    cost: 822.296377ms, prompt: [he], inference: [ po]
    cost: 982.226678ms, prompt: [ poi], inference: [rier]
    cost: 1.000032715s, prompt: [vr], inference: [`]
    cost: 840.470374ms, prompt: [ée], inference: [
    ]
    cost: 832.269423ms, prompt: [
    ], inference: [ch]
    cost: 834.183225ms, prompt: [pl], inference: [um]
    cost: 844.795164ms, prompt: [ush], inference: [ =>]
    cost: 1.191078351s, prompt: [ gir], inference: [ld]
    cost: 855.975299ms, prompt: [af], inference: [ =>]
    cost: 1.235930318s, prompt: [e], inference: [ =>]
    cost: 850.870609ms, prompt: [ =>], inference: [ j]
    cost: 1.196573672s, prompt: [ gir], inference: [ou]
    cost: 848.317854ms, prompt: [af], inference: [ de]
    cost: 1.243230272s, prompt: [e], inference: [ de]
    cost: 894.087427ms, prompt: [ pel], inference: [uche]
    cost: 1.243929315s, prompt: [uche], inference: [
    ]
    cost: 822.039795ms, prompt: [
    ], inference: [st]
    cost: 847.290121ms, prompt: [che], inference: [ese]
    cost: 840.599632ms, prompt: [ese], inference: [ c]
    cost: 831.24577ms, prompt: [ =>], inference: [ from]
    cost: 842.62046ms, inference: [age]
    cost: 828.848962ms, inference: [
    ]
    cost: 823.01183ms, inference: [b]
    cost: 1.245383078s, inference: [read]
    cost: 849.658011ms, inference: [ =>]
    cost: 855.406345ms, inference: [ pain]
    cost: 916.182704ms, inference: [
    ]
    cost: 829.437727ms, inference: [h]
    cost: 1.236134944s, inference: [amb]
    cost: 837.068272ms, inference: [urger]
    cost: 1.182976401s, inference: [ =>]
    cost: 848.510569ms, inference: [ h]
    cost: 860.437766ms, inference: [amb]
    cost: 843.537497ms, inference: [urger]
    cost: 1.190157387s, inference: [
    ]
    cost: 830.706305ms, inference: [s]
    Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese => fromage
    bread => pain
    hamburger => hamburger
    s
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

## cluster computing

cluster computing will implement in the feature