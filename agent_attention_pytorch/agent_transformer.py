import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class AgentSelfAttention(Module):
    def __init__(
        self,
        dim,
        *,
        num_agent_tokens,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        talking_heads = True,
        gate = True,
        sub_layernorm = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if gate else None

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.ak_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

        self.qa_dropout = nn.Dropout(dropout)
        self.ak_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_head) if sub_layernorm else nn.Identity(),
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        *,
        agent_tokens,
        mask = None,
        return_agent_tokens = False
    ):
        x = self.norm(x)
        a = self.norm(agent_tokens)

        x_and_agents, xa_ps = pack([a, x], 'b * d')
        qkv = self.to_qkv(x_and_agents)

        qkv_agent, qkv_input = unpack(qkv, xa_ps, 'qkv b h * d')

        q, k, v = qkv_input
        agent_queries, agent_keys, _ = qkv_agent

        q = q * self.scale
        agent_queries = agent_queries * self.scale

        qa_sim = einsum('b h i d, b h j d -> b h i j', q, agent_keys)
        ak_sim = einsum('b h i d, b h j d -> b h i j', agent_queries, k)

        if exists(mask):
            max_neg_value = -torch.finfo(qa_sim.dtype).max
            ak_sim = ak_sim.masked_fill(~rearrange(mask, 'b j -> b 1 1 j'), max_neg_value)

        qa_attn = qa_sim.softmax(dim = -1)
        ak_attn = ak_sim.softmax(dim = -1)

        qa_attn = self.qa_dropout(qa_attn)
        ak_attn = self.ak_dropout(ak_attn)

        qa_attn = self.qa_talking_heads(qa_attn)
        ak_attn = self.ak_talking_heads(ak_attn)

        agent_out = einsum('b h i j, b h j d -> b h i d', ak_attn, v)

        out = einsum('b h i j, b h j d -> b h i d', qa_attn, agent_out)

        if exists(mask):
            out = out.masked_fill(~rearrange(mask, 'b n -> b 1 n 1'), 0.)

        if exists(self.to_gates):
            out = out * self.to_gates(x)
            agent_out = agent_out * self.to_gates(a)

        out = self.to_out(out)
        agent_out = self.to_out(agent_out)

        if not return_agent_tokens:
            return out

        return out, agent_out

# transformer

class AgentTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_agent_tokens,
        depth,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        final_norm = True,
        **attn_kwargs: dict
    ):
        super().__init__()

        self.agent_tokens = nn.Parameter(torch.zeros(num_agent_tokens, dim))
        nn.init.normal_(self.agent_tokens, std = 0.02)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                AgentSelfAttention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    num_agent_tokens = num_agent_tokens,
                    **attn_kwargs
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.final_norm = RMSNorm(dim) if final_norm else None

    def forward(
        self,
        x,
        mask = None,
        return_agent_tokens = False
    ):
        batch = x.shape[0]
        a = repeat(self.agent_tokens, 'm d -> b m d', b = batch)

        for attn, ff in self.layers:
            attn_out, agent_out = attn(
                x,
                agent_tokens = a,
                mask = mask
            )

            a = a + agent_out
            x = x + attn_out

            x, ps = pack([a, x], 'b * d')
        
            x = ff(x) + x

            a, x = unpack(x, ps, 'b * d')

        if exists(self.final_norm):
            x = self.final_norm(x)
            a = self.final_norm(a)

        if not return_agent_tokens:
            return x

        return x, a
