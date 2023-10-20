# llama2.go

Port of Facebook's LLaMA model in pure go and use little memory.

## memory usage

| Model | Precision | Memory | Memory(Cached Params) |
| ----- | --------- | ------ | --------------------- |
| 7B  | bf16 | 600M+ | 17G+ |
| 13B | bf16 | 1G+ | 32G+ |
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
    2023/10/20 17:02:36 [INFO]model loaded
    2023/10/20 17:02:36 [INFO]model info:
    2023/10/20 17:02:36 [INFO]  + embedding dim: 4096
    2023/10/20 17:02:36 [INFO]  + layers: 32
    2023/10/20 17:02:36 [INFO]  + heads: 32
    2023/10/20 17:02:36 [INFO]  + kv_heads: 32
    2023/10/20 17:02:36 [INFO]  + q_embedding each head: 128
    2023/10/20 17:02:36 [INFO]  + kv_embedding each head: 128
    2023/10/20 17:02:36 [INFO]  + norm_eps: 1.000000e-05
    2023/10/20 17:02:36 [INFO]warm up model...
    2023/10/20 17:02:43 [INFO]warm up done
    cost: 805.096668ms, prompt: [<s>], inference: [ The]
    cost: 794.034869ms, prompt: [Tra], inference: [vel]
    cost: 791.610401ms, prompt: [ns], inference: [it]
    cost: 785.028969ms, prompt: [late], inference: [.]
    cost: 797.443303ms, prompt: [ Eng], inference: [
    ]
    cost: 789.511594ms, prompt: [lish], inference: [ to]
    cost: 795.109561ms, prompt: [ to], inference: [ H]
    cost: 786.051182ms, prompt: [ Fre], inference: [ch]
    cost: 785.656581ms, prompt: [nc], inference: [sh]
    cost: 802.650179ms, prompt: [h], inference: [
    ]
    cost: 789.006086ms, prompt: [:], inference: [
    ]
    cost: 785.612863ms, prompt: [
    ], inference: [What]
    cost: 790.960297ms, prompt: [
    ], inference: [
    ]
    cost: 788.613766ms, prompt: [se], inference: [ek]
    cost: 793.616298ms, prompt: [a], inference: [
    ]
    cost: 788.519639ms, prompt: [ ott], inference: [om]
    cost: 786.268165ms, prompt: [er], inference: [
    ]
    cost: 799.88383ms, prompt: [ =>], inference: [
    ]
    cost: 790.308092ms, prompt: [ lo], inference: [b]
    cost: 789.359057ms, prompt: [ut], inference: [ier]
    cost: 792.335992ms, prompt: [re], inference: [
    ]
    cost: 799.485639ms, prompt: [ de], inference: [ la]
    cost: 790.988376ms, prompt: [ mer], inference: [
    ]
    cost: 792.840929ms, prompt: [
    ], inference: [
    ]
    cost: 789.421174ms, prompt: [pe], inference: [as]
    cost: 796.805358ms, prompt: [pper], inference: [ =>]
    cost: 797.496379ms, prompt: [min], inference: [ =>]
    cost: 799.876148ms, prompt: [t], inference: [ =>]
    cost: 795.272906ms, prompt: [ =>], inference: [ po]
    cost: 800.170421ms, prompt: [ ment], inference: [he]
    cost: 793.94568ms, prompt: [he], inference: [ po]
    cost: 798.776807ms, prompt: [ poi], inference: [il]
    cost: 792.626643ms, prompt: [vr], inference: [ée]
    cost: 793.800323ms, prompt: [ée], inference: [
    ]
    cost: 802.375475ms, prompt: [
    ], inference: [
    ]
    cost: 792.565524ms, prompt: [pl], inference: [aster]
    cost: 802.16297ms, prompt: [ush], inference: [ =>]
    cost: 802.694426ms, prompt: [ gir], inference: [ll]
    cost: 792.011098ms, prompt: [af], inference: [ =>]
    cost: 794.957684ms, prompt: [e], inference: [ =>]
    cost: 793.161168ms, prompt: [ =>], inference: [ gir]
    cost: 808.839479ms, prompt: [ gir], inference: [raf]
    cost: 793.328555ms, prompt: [af], inference: [ou]
    cost: 798.369926ms, prompt: [e], inference: [ dou]
    cost: 799.757471ms, prompt: [ pel], inference: [uche]
    cost: 791.91408ms, prompt: [uche], inference: [
    ]
    cost: 795.313464ms, prompt: [
    ], inference: [
    ]
    cost: 802.494011ms, prompt: [che], inference: [ese]
    cost: 801.804049ms, prompt: [ese], inference: [ c]
    cost: 799.417178ms, prompt: [ =>], inference: [ from]
    cost: 794.279909ms, inference: [age]
    cost: 794.393727ms, inference: [
    ]
    cost: 800.798864ms, inference: [
    ]
    cost: 803.566172ms, inference: [Tra]
    cost: 805.789997ms, inference: [ans]
    cost: 798.737076ms, inference: [late]
    cost: 801.030808ms, inference: [ Fre]
    cost: 801.655272ms, inference: [nc]
    cost: 798.778104ms, inference: [ch]
    cost: 807.62774ms, inference: [ to]
    cost: 800.958739ms, inference: [ English]
    cost: 798.907121ms, inference: [:]
    cost: 796.359556ms, inference: [
    ]
    cost: 807.457454ms, inference: [
    ]
    cost: 810.997564ms, inference: [lo]
    cost: 803.872655ms, inference: [ut]
    =====================================
    Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese => fromage

    Traanslate Frencch to English:

    lout
    2023/10/20 17:03:35 [INFO]total cost: 52.582370596s, 1.27token/s
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