"""Independent Alpha/response-route diagnostic; production renderer stays untouched."""
import numpy as np
import torch
from torch.nn import functional as F

from response_conditioned_rgb_common import ResponseConditionedDecoder


class DiagnosticResponseDecoder(ResponseConditionedDecoder):
    def __init__(self, owner, epsilon, *, input_alpha=True, film_alpha=True,
                 projection=True, ca_response=True, parameterization='hard'):
        super().__init__(owner, epsilon)
        self.input_alpha, self.film_alpha = bool(input_alpha), bool(film_alpha)
        self.use_projection, self.use_ca_response = bool(projection), bool(ca_response)
        if parameterization not in ('hard', 'smooth'):
            raise ValueError(parameterization)
        self.parameterization = parameterization
        self.capture_pixels = False
        self.pixel_tensors = {}
        if not projection:
            self.response_projection.requires_grad_(False)
        if not ca_response:
            for block in owner.backbone.cross_attention:
                block.cross_attention.requires_grad_(False)
                block.image_norm.requires_grad_(False)
                block.wrench_norm.requires_grad_(False)

    def forward(self, images, response, spatial_gates, alpha, *, trace=False):
        b, steps = next(iter(images.values())).shape[:2]
        if response.shape != (b, steps*len(self.owner.image_keys)*196, 256):
            raise ValueError('Response layout must be time, configured camera, row-major14x14, channel256')
        raw_response = response.reshape(b, steps, len(self.owner.image_keys), 196, 256)
        response = raw_response*alpha[..., None, None, None] if self.input_alpha else raw_response
        stats, camera_stats = {}, {}
        def measure(name, value):
            if trace:
                energy = value.detach().float().reshape(b, -1).square().mean(-1)
                camera_stats.setdefault(name, []).append(energy)
        if trace:
            stats = dict(alpha=alpha.detach(), alpha_input=alpha.detach() if self.input_alpha else torch.ones_like(alpha),
                alpha_film=alpha.detach() if self.film_alpha else torch.ones_like(alpha), alpha_rgb=alpha.detach(),
                response_raw_mse=raw_response.float().flatten(1).square().mean(-1),
                response_input_mse=response.float().flatten(1).square().mean(-1))
        result, backbone = {}, self.owner.backbone
        for camera, key in enumerate(self.owner.image_keys):
            image, gate = images[key], spatial_gates[key].flatten(0, 1)
            condition = response[:, :, camera].reshape(b*steps, 196, 256)
            embedding = self.owner.camera_embedding.weight[camera].reshape(1, 1, -1)
            flat = image.flatten(0, 1)
            height, width = flat.shape[-2:]
            p = backbone.patch_size
            padded = F.pad(flat, (0,(-width)%p,0,(-height)%p), mode='replicate')
            features, skips = backbone.stem(padded), []
            for encoder in backbone.encoders:
                skips.append(features);features=encoder(F.avg_pool2d(features,2))
            gh,gw=features.shape[-2:]
            tokens=features.flatten(2).transpose(1,2)
            tokens=tokens+backbone._position(gh,gw,tokens)+embedding.to(tokens)
            for group,cross in enumerate(backbone.cross_attention):
                for block in backbone.image_self_attention[2*group:2*group+2]:tokens=block(tokens)
                measure(f'ca{group}_image_before_mse',tokens)
                if self.use_ca_response:
                    if trace:
                        q,k=cross.image_norm(tokens),cross.wrench_norm(condition)
                        update,_=cross.cross_attention(query=q,key=k,value=condition,need_weights=False)
                        measure(f'ca{group}_key_normalized_mse',k)
                        measure(f'ca{group}_value_mse',condition)
                        measure(f'ca{group}_attention_update_mse',update)
                        tokens=tokens+cross.dropout(update)
                        ffn=cross.ffn(cross.output_norm(tokens))
                        measure(f'ca{group}_ffn_update_mse',ffn)
                        tokens=tokens+ffn
                    else:tokens=cross(tokens,condition)
                else:
                    # Preserve the pure-visual FFN so C isolates response reading.
                    ffn=cross.ffn(cross.output_norm(tokens))
                    measure(f'ca{group}_ffn_update_mse',ffn)
                    tokens=tokens+ffn
                measure(f'ca{group}_image_after_mse',tokens)
            features=backbone.norm(tokens).transpose(1,2).reshape(b*steps,backbone.embed_dim,gh,gw)
            measure('projection_feature_before_mse',features)
            if self.use_projection:
                update=self.response_projection(condition.transpose(1,2).reshape(b*steps,256,14,14))
                measure('projection_update_mse',update)
                features=features+F.interpolate(update,size=features.shape[-2:],mode='bilinear',align_corners=False)
            measure('projection_feature_after_mse',features)
            fused=features
            pooled=backbone.decoder_film.pool(condition)
            measure('film_pooled_condition_mse',pooled)
            for index,(decoder,skip) in enumerate(zip(backbone.decoders,reversed(skips))):
                if backbone.conv_block=='bottleneck':features=decoder(features,skip)
                else:features=decoder(torch.cat([F.interpolate(features,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
                if index<len(self.owner.spatial_modulations):
                    before=features
                    measure(f'spatial{index}_feature_before_mse',features)
                    features=self.owner.spatial_modulations[index](features,fused,gate)
                    measure(f'spatial{index}_feature_after_mse',features)
                    measure(f'spatial{index}_applied_update_mse',features-before)
                if index<2:
                    coefficient=alpha.flatten() if self.film_alpha else torch.ones_like(alpha.flatten())
                    if trace:
                        module=backbone.decoder_film
                        gamma,beta=module.affines[index](pooled).chunk(2,-1)
                        spatial=F.interpolate(gate.to(features),size=features.shape[-2:],mode='bilinear',align_corners=False)
                        raw=gamma[...,None,None]*features+beta[...,None,None]
                        measure(f'film{index}_feature_before_mse',features)
                        measure(f'film{index}_update_raw_mse',raw)
                        measure(f'film{index}_after_gate_mse',spatial*raw)
                        measure(f'film{index}_after_alpha_mse',coefficient.to(features).reshape(-1,1,1,1)*spatial*raw)
                    features=backbone.decoder_film(features,pooled,index,gate,coefficient)
                    measure(f'film{index}_feature_after_mse',features)
            features=F.interpolate(features,size=(height,width),mode='bilinear',align_corners=False)
            # Match the original BF16 tanh, then cast to FP32 for RGB arithmetic.
            logits=backbone.output(features).reshape_as(image)
            raw=logits.tanh().float()
            measure('rgb_raw_mse',raw)
            measure('rgb_after_scale_mse',self.owner.residual_scale*raw)
            measure('rgb_after_spatial_gate_mse',self.owner.residual_scale*raw*spatial_gates[key].float())
            support=spatial_gates[key].float()*alpha[...,None,None,None]
            delta=(self.owner.residual_scale if self.parameterization=='hard' else self.epsilon)*raw*support
            measure('rgb_after_alpha_mse',delta)
            bound=self.epsilon*support
            prebound=delta
            if self.parameterization=='hard':
                delta=torch.maximum(-bound,torch.minimum(bound,delta))
            measure('rgb_after_bound_mse',delta)
            unbounded_image=image.float()+delta
            result[key]=torch.where(support>0,unbounded_image.clamp(0,1),image.float())
            measure('rgb_after_clip_mse',result[key]-image.float())
            if trace:
                allowed=(support>0).expand_as(raw)
                hard=(prebound.abs()>bound.expand_as(raw)) & allowed if self.parameterization=='hard' else torch.zeros_like(allowed)
                range_clip=((unbounded_image<0)|(unbounded_image>1)) & allowed
                slope=(self.owner.residual_scale if self.parameterization=='hard' else self.epsilon)*support*(1-raw.square())
                slope=slope.masked_fill(hard|range_clip|~allowed,0)
                count=lambda value:value.flatten(1).sum(-1).float()
                for name,value in dict(allowed_channels=count(allowed),total_channels=torch.full((b,),raw[0].numel(),device=raw.device,dtype=torch.float32),
                    hard_clipped_channels=count(hard),range_clipped_channels=count(range_clip),
                    effective_channels=count((slope.abs()>1e-8)&allowed),
                    relative_effective_channels=count((slope.abs()>1e-3*self.epsilon*support)&allowed),
                    support_sum=support.flatten(1).sum(-1),support_count=torch.full((b,),support[0].numel(),device=raw.device,dtype=torch.float32)).items():
                    camera_stats.setdefault(name,[]).append(value)
                measure('pixel_local_slope_mse',slope)
            if self.capture_pixels:
                self.pixel_tensors[key]=dict(logits=logits,output=result[key],support=support)
                for tensor in (logits,result[key]):
                    if tensor.requires_grad:tensor.retain_grad()
        if trace:
            stats.update({key:torch.stack(values).mean(0) for key,values in camera_stats.items()})
            return result,stats
        return result


def summarize_traces(rows):
    def summary(selected):
        if not selected:return dict(samples=0)
        keys=set.intersection(*(set(row) for row in selected))-{'index','episode','phase'}
        values={key:np.concatenate([np.asarray(row[key]).reshape(-1) for row in selected]) for key in keys}
        return dict(samples=len(selected),rms={key.removesuffix('_mse'):float(np.sqrt(v.mean()))
                    for key,v in values.items() if key.endswith('_mse')},
            alpha={key:dict(mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),
                p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)))
                for key,v in values.items() if key.startswith('alpha')})
    return dict(all=summary(rows),phases={name:summary([r for r in rows if r['phase']==p])
        for p,name in enumerate(['free','precontact','contact'])},
        episodes={str(ep):summary([r for r in rows if r['episode']==ep]) for ep in sorted({r['episode'] for r in rows})})
