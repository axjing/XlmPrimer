import json
from dataclasses import dataclass,field,asdict

@dataclass
class LLMConfig:
    
    # attribute_map:dict={
    #     "hidden_size": "n_embd",
    #     "max_position_embeddings": "n_positions",
    #     "num_attention_heads": "n_head",
    #     "num_hidden_layers": "n_layer",
    # }
    
    lm_model_type = "GPT-2"
    # model scale parameters
    n_positions:int=1024 # 最大上下文长度
    max_length=512
    vocab_size:int=50304 # GPT-2的词汇表大小为50257，为提高效率，填充到最接近的64的倍数。
    
    n_embd:int=768 # word embedding/hidden 维度
    n_layers:int=12 # Transformer块 层数
    n_heads:int=12 # 多头注意力头数，n_embd // n_head= 64，每个头的维度为64
    n_kv_heads:int=2
    n_intermediate :int=4*n_embd # 前馈网络中间层维度
    
    # activation & normalization parameters
    activation_function:str="relu"
    dropout:float=0.0 # dropout 率，用于防止过拟合
    resid_pdrop:float=0.1 # 残差链接层 dropout 率，用于防止过拟合
    embd_pdrop:float=0.1 # embedding 层 dropout 率，用于防止过拟合
    attn_pdrop:float=0.1 # 多头注意力层 dropout 率，用于防止过拟合
    layer_norm_epsilon:float=1e-5 # LayerNorm 中的 epsilon 值，用于防止除0错误
    bias:bool=False # True：线性层和 LayerNorm 中存在偏置，就像 GPT-2 一样。False：性能稍好且速度更快
    
    # generation & pool
    summary_type="cls_index" # 取<CLS> 位置做句子、分类表示
    summary_use_proj:bool=True # 是否对池化结果做线性投影
    summary_activation:str|None=None # 池化结果的激活函数，默认使用 None
    summary_proj_to_labels:bool=True # 是否对池化结果做线性投影到标签维度
    summary_first_dropout:float=0.1 # 池化结果的 dropout 率，用于防止过拟合
    
    
    # attention & inference 优化 parameters
    use_cache:bool=True
    scale_attn_weights: bool = True # 注意力分数按头维度的根号缩放
    scale_attn_by_inverse_layer_idx:bool=True # 是否按层索引的倒数缩放注意力权重
    reorder_and_upcast_attn=False # 不重排、提升精度计算注意力    
    
    
    # special tokens
    bos_token_id=5026 # 开始符 token id，用于表示序列的开始
    eos_token_id=5026 # 介绍符 token id,用于表示序列的结束,默认与开始符相同
    pad_token_id=None # 填充 token id，用于填充到最大上下文长度
    
    # 结构开关
    add_cross_attention:bool=False # 是否添加跨注意力机制
    tie_word_embeddings=True # 是否将 word embedding 层和 output 层的权重共享(GPT经典做法，为了省参数量)
    
    
    rotary_emb_base:float= 100000 # 旋转位置编码基数
    attn_scaling:float=1.0 # 注意力机制缩放
    
    lm_use_tokens:bool=False # 判断该大语言模型的输入形式为token ids还是嵌入向量（若作为视觉语言模型的主干网络，则设为否）。输入张量。若`lm_use_tokens`为真，该参数需传入形状为(bsz,seq_len)的token索引；若为假，则需传入形状为(bsz,seq_len,n_embd)的嵌入向量。
    lm_tie_weights:bool=True # 决定是否将语言模型输出层权重与词嵌入权重进行绑定
    lm_tokenizer:str="HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template:str="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    
    def to_json(file_path:str=None,indent:int=4):
      """
      配置转JSON：可返回字符串 / 直接写入文件
      :param file_path: 保存的文件路径，为None时仅返回字符串
      :param indent: 格式化缩进
      :return: file_path为None 返回JSON字符串；写入文件则返回None
      """
      
      data_dict=asdict(self)
      json_str=json.dumps(data_dict,ensure_ascii=False,indent=indent)
      
      if file_path:
        with open(file_path,'w',encoding='utf-8') as f:
          json.dump(data_dict,f,ensure_ascii=False,indent=indent)
        print(f'>>> Json saved: {file_path}')
      
    
    

    
@dataclass
class VLMConfig(LLMConfig):
    # Language
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    n_positions: int = 4096
    max_length:int=4096
    
    n_embd: int = 960
    n_intermediate: int = 2560
    n_layers: int = 32
    n_heads: int = 15
    n_kv_heads:int=5
    
    
    activation_function: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attn_pdrop: float | int = 0.0
    # This differs from `CLIPTokenizer`'s default and from openai/siglip
    # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407
    projection_size: int | None = None
    
    # Vsion
    vit_model_type = "google/siglip2-base-patch16-512"
    
    n_channels: int = 3
    image_size: int = 512
    patch_size: int = 16
    
    vit_pdropout:float=0.0
    
    
    vit_n_embd:int=768
    vit_n_intermediate:int = 4*vit_n_embd
    vit_n_heads:int=12
    vit_n_layers:int=12
    vit_layernorm_eps:float=1e-6
    vit_cls_flag: bool = False
    
    mp_pixel_shuffle_factor:int=4
    mp_image_token_length: int = 64
    
    max_img_size: int = 2048
    resize_to_max_side_len: bool = True
    
    vlm_load_backbone_weights: bool = True
    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    
    vlm_checkpoint_path:str='checkpoints'
    hf_repo_name:str="VLM"
    
    