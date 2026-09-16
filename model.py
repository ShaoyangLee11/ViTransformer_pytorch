import torch
import torch.nn as nn
import math

"""
ViT is an Encoder-only Architecture
"""

class LayerNormalization(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()

        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))
        self.eps=eps

    def forward(self,x:torch.Tensor)->torch.Tensor:
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)

        return (self.alpha*(x-mean)/std+self.eps)+self.beta



class PatchEmbedding(nn.Module):
    def __init__(self,img_size:int=224,patch_size:int=16,in_chans:int=3,embed_dim=768,flatten=True):
        super().__init__()
        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size)

        self.img_size=img_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim

        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.patch_num=self.grid_size[0]*self.grid_size[1]
        self.flatten=flatten

        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size) #(B,C,H,W)-->(B,e_d,grid_size[0],grid_size[1])
        self.norm=LayerNormalization()

        self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))


    def forward(self,x:torch.Tensor):
        B,_,H,W=x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x=self.proj(x)
        if self.flatten:
            x=x.reshape(B,self.embed_dim,-1).transpose(1,2)      #(B,C,H,W)-->(B,e_d,grid_size[0],grid_size[1])-->(B,e_d,p_n)-->(B,p_n,e_d)
            cls_token = self.cls_token.expand(B, -1, -1)
            x=torch.cat([cls_token,x],dim=1) # (B,p_n,e_d)-->(B,patch_num+1,e_d)

        return self.norm(x)

# (B,patch_num,embed_dim)  or  (B,patch_num,d_model)


# of course u can oversee this part because timm did not use this unlearnable PE,
class PositionalEncoding(nn.Module):
    def __init__(self,patch_num,d_model=768,dropout:float=0.1):
        super().__init__()
        self.patch_num=patch_num
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)

        pe=torch.ones(size=(patch_num+1,d_model))                 # (patch_num,d_model)
        pos=torch.arange(0,patch_num+1,step=1).unsqueeze(dim=1)   # (patch_num,1)
        """
        pos是token的位置,在此处是patch的相对位置
        """
        div_term=torch.exp(torch.arange(0,d_model,step=2).float()*(-math.log(10000.0))/d_model)  # (d_model/2,)

        """
        pos的维度是(patch_num,1),div_term的维度(d_model_2,)

        相乘之前先进行广播(这是因为两者无法逐元素相乘,因此必须广播):
        
        (patch_num,1)       (patch_num,1)       (patch_num,d_model/2)   [重复d_model/2遍]
                        -->                 -->                         -->之后逐元素相乘
        (d_model/2,)        (1,d_model/2)       (patch_num,d_model/2)   [重复patch_num遍] 
        
        """
        odd_part=torch.sin(pos*div_term)                    # (patch_num,d_model/2) 
        even_part=torch.cos(pos*div_term)                   # (patch_num,d_model/2) 

        pe[:,0::2]=even_part
        pe[:,1::2]=odd_part
        self.register_buffer('pe',pe)                       # (patch_num,d_model)
        

    def forward(self,x:torch.Tensor):

        x=x+self.pe

        return self.dropout(x)                              # (B,patch_num,d_model)



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int=768,h:int=12,dropout:float=0.1):
        super().__init__()

        self.h=h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

        self.d_k=d_model//h
        assert d_model%h==0,"Number of heads is not allowed"

        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    @staticmethod
    def calc_attention(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor):
        d_k=query.shape[-1]

        # (B,h,patch_num,d_k)-->(B,h,patch_num,patch_num)
        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)
        attention_scores=attention_scores.softmax(dim=-1)

        return (attention_scores@value),attention_scores


    def forward(self,q,k,v):
        query=self.w_q(q)                                   # (B,patch_num,d_k*h)
        key=self.w_k(k)
        value=self.w_v(v)

        # Actually following part is needless

        # (B,patch_num,d_k*h)-->(B,patch_num,h,d_k)-->(B,h,patch_num,d_k)
        query=query.reshape(query.shape[0],query.shape[1],self.h,-1).transpose(1,2)
        key=key.reshape(key.shape[0],key.shape[1],self.h,-1).transpose(1,2)
        value=value.reshape(value.shape[0],value.shape[1],self.h,-1).transpose(1,2)

        x,_=MultiHeadAttention.calc_attention(query,key,value)  # (B,h,patch_num,d_k)

        x=x.transpose(1,2).reshape(x.shape[0],-1,self.h*self.d_k)

        return self.norm(self.dropout(self.w_o(x)))

class ResidualBlock(nn.Module):
    def __init__(self,dropout:float=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        x=x+sublayer(x)
        x=self.norm(x)

        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, d_model:int=768,dropout:float=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_model*4)
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_model*4,d_model)

    def forward(self,x)->torch.Tensor:
        x=self.dropout(self.gelu(self.linear1(x)))

        return self.dropout(self.linear2(x))


class EncoderBlock(nn.Module):

    def __init__(self,d_model=768,h=12):
        super().__init__()

        self.multi_head_attn=MultiHeadAttention(d_model,h)
        self.mlp=MLP(d_model)
        self.residual_block=nn.ModuleList([ResidualBlock() for _ in range(2)])


    def forward(self,x):
       
        x=self.residual_block[0](x,lambda x:self.multi_head_attn(x,x,x))
        x=self.residual_block[1](x,lambda x:self.mlp(x))

        return x


class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers

    def forward(self,x):

        for layer in self.layers:
            x=layer(x)

        return x

# (B,P,d_model)

class ProjectLayer(nn.Module):
    def __init__(self,d_model,classification):
        super().__init__()
        self.proj=nn.Linear(d_model,classification)

    def forward(self,x:torch.Tensor):
        cls_token_output=x[:,0,:]
        x=self.proj(cls_token_output).squeeze()

        return x
    


class ViTransformer(nn.Module):

    def __init__(self,patch_embd:PatchEmbedding,pos_enc:PositionalEncoding,encoder:Encoder,mlp:MLP,proj:ProjectLayer):
        super().__init__()
        self.patch_embd=patch_embd
        self.pos_enc=pos_enc
        self.encoder=encoder
        self.mlp=mlp
        self.proj=proj

    def forward(self,x):

        x=self.pos_enc(self.patch_embd(x))

        x=self.encoder(x)

        x=self.mlp(x)

        return self.proj(x)


def build_ViTransformer(img_size,patch_size,in_chans,d_model,classification,N:int=8,h:int=12,dropout:float=0.1):

    patch_embd=PatchEmbedding(img_size,patch_size,in_chans,d_model)
    patch_num=(img_size//patch_size)*(img_size//patch_size)
    pos_enc=PositionalEncoding(patch_num,d_model,dropout)

    encoder_list=[]

    for _ in range(N):
        encoder_block=EncoderBlock(d_model,h)
        encoder_list.append(encoder_block)

    encoder=Encoder(nn.ModuleList(encoder_list))

    mlp_layer=MLP(d_model,dropout)

    proj_layer=ProjectLayer(d_model,classification)

    viTransformer=ViTransformer(patch_embd,pos_enc,encoder,mlp_layer,proj_layer)


    return viTransformer




        



        






