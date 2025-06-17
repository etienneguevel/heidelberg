import torch
import types

def custom_forward(self, x, return_attn=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    if return_attn:
        return attn

    return x

def get_attn(img, model, layer):

    img = img.unsqueeze(0)
    with torch.no_grad():
        patches = model.get_intermediate_layers(img, return_prefix_tokens=True, n=2)

    img_inter, cls_inter = patches[0]
    cls_token = torch.cat([cls_inter, img_inter], dim=1)
    
    with torch.no_grad():
        # Modifiy the forward method to also get the attention map
        layer.attn.forward = types.MethodType(custom_forward, layer.attn)
        cls_attn = layer.attn.forward(x=cls_token, return_attn=True)[0, :, 0, 1:]
    return cls_attn