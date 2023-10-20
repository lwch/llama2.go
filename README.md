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
    2023/10/20 16:53:33 [INFO]model loaded
    2023/10/20 16:53:33 [INFO]model info:
    2023/10/20 16:53:33 [INFO]  + embedding dim: 4096
    2023/10/20 16:53:33 [INFO]  + layers: 32
    2023/10/20 16:53:33 [INFO]  + heads: 32
    2023/10/20 16:53:33 [INFO]  + kv_heads: 32
    2023/10/20 16:53:33 [INFO]  + q_embedding each head: 128
    2023/10/20 16:53:33 [INFO]  + kv_embedding each head: 128
    2023/10/20 16:53:33 [INFO]  + norm_eps: 1.000000e-05
    2023/10/20 16:53:33 [INFO]warm up model...
    2023/10/20 16:53:40 [INFO]warm up done
    cost: 829.810388ms, prompt: [<s>], inference: [ Home]
    cost: 812.214704ms, prompt: [Tra], inference: [vel]
    cost: 806.84939ms, prompt: [ns], inference: [parent]
    cost: 812.460918ms, prompt: [late], inference: [.]
    cost: 812.157835ms, prompt: [ Eng], inference: [
    ]
    cost: 814.588164ms, prompt: [lish], inference: [ to]
    cost: 818.31183ms, prompt: [ to], inference: [ Ur]
    cost: 810.059738ms, prompt: [ Fre], inference: [ch]
    cost: 807.891792ms, prompt: [nc], inference: [sh]
    cost: 820.362737ms, prompt: [h], inference: [
    ]
    cost: 809.772945ms, prompt: [:], inference: [
    ]
    cost: 834.103258ms, prompt: [
    ], inference: [Language]
    cost: 824.401814ms, prompt: [
    ], inference: [https]
    cost: 813.738073ms, prompt: [se], inference: [ek]
    cost: 815.641784ms, prompt: [a], inference: [
    ]
    cost: 823.850247ms, prompt: [ ott], inference: [oman]
    cost: 815.05011ms, prompt: [er], inference: [
    ]
    cost: 836.127635ms, prompt: [ =>], inference: [ sea]
    cost: 819.677082ms, prompt: [ lo], inference: [b]
    cost: 819.233433ms, prompt: [ut], inference: [
    ]
    cost: 811.388681ms, prompt: [re], inference: [
    ]
    cost: 821.7185ms, prompt: [ de], inference: [ la]
    cost: 816.537101ms, prompt: [ mer], inference: [
    ]
    cost: 816.128931ms, prompt: [
    ], inference: [
    ]
    cost: 825.586391ms, prompt: [pe], inference: [asant]
    cost: 818.89321ms, prompt: [pper], inference: [ =>]
    cost: 820.790231ms, prompt: [min], inference: [st]
    cost: 815.140905ms, prompt: [t], inference: [ =>]
    cost: 815.614272ms, prompt: [ =>], inference: [ po]
    cost: 823.918492ms, prompt: [ ment], inference: [he]
    cost: 840.859783ms, prompt: [he], inference: [ po]
    cost: 830.357498ms, prompt: [ poi], inference: [rier]
    cost: 818.728013ms, prompt: [vr], inference: [ée]
    cost: 826.150385ms, prompt: [ée], inference: [
    ]
    cost: 822.37227ms, prompt: [
    ], inference: [
    ]
    cost: 813.261271ms, prompt: [pl], inference: [um]
    cost: 826.634082ms, prompt: [ush], inference: [ =>]
    cost: 829.685964ms, prompt: [ gir], inference: [der]
    cost: 817.732717ms, prompt: [af], inference: [ =>]
    cost: 830.508462ms, prompt: [e], inference: [ =>]
    cost: 819.823964ms, prompt: [ =>], inference: [ g]
    cost: 820.070487ms, prompt: [ gir], inference: [raf]
    cost: 822.022788ms, prompt: [af], inference: [es]
    cost: 833.944654ms, prompt: [e], inference: [ m]
    cost: 811.898174ms, prompt: [ pel], inference: [uche]
    cost: 819.079282ms, prompt: [uche], inference: [
    ]
    cost: 818.619303ms, prompt: [
    ], inference: [
    ]
    cost: 821.409612ms, prompt: [che], inference: [ese]
    cost: 821.450601ms, prompt: [ese], inference: [ c]
    cost: 828.624458ms, prompt: [ =>], inference: [ from]
    cost: 815.680186ms, inference: [age]
    cost: 826.836083ms, inference: [
    ]
    cost: 830.086712ms, inference: [
    ]
    cost: 831.248282ms, inference: [Tra]
    cost: 825.719731ms, inference: [ans]
    cost: 828.722695ms, inference: [late]
    cost: 825.164305ms, inference: [ French]
    cost: 819.44112ms, inference: [ to]
    cost: 825.693212ms, inference: [ English]
    cost: 841.11027ms, inference: [:]
    cost: 840.032918ms, inference: [
    ]
    cost: 826.969138ms, inference: [
    ]
    cost: 825.815217ms, inference: [s]
    cost: 820.554111ms, inference: [alt]
    cost: 822.608378ms, inference: [ =>]
    cost: 821.546985ms, inference: [ sel]
    =====================================
    Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese => fromage

    Traanslate French to English:

    salt => sel
    2023/10/20 16:54:34 [INFO]total cost: 54.26254411s, 1.23token/s
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