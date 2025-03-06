from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 8,
            tie_word_embeddings: bool = True,
            ###########################################
            attention:str='GQA',
            #GQA
            n_heads: int = 14,
            n_kv_heads: int = 2,
            #MLA
            q_lora_rank: int=0,
            kv_lora_rank: int=512,
            qk_nope_head_dim: int=64,
            qk_rope_head_dim:int=64,
            v_head_dim:int=64,
            #############################################
            vocab_size: int = 151650,
            # vocab_size: int = 6400,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 8192,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            ####################################################
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        #####################################################
        self.attention=attention
        #GQA
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        #MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)
