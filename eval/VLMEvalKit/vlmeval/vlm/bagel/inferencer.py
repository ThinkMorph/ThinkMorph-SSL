# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from .data.data_utils import pil_img2rgb
from .modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        vae_param = next(self.vae_model.parameters())
        self.vae_device = vae_param.device
        self.vae_dtype = vae_param.dtype
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
        noise_seed=None,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
            noise_seed=noise_seed,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        latent = latent.to(
            device=self.vae_device,
            dtype=self.vae_dtype,
            non_blocking=True,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    def batch_decode_images(self, latents, image_shapes):
        Hs, Ws = [s[0] for s in image_shapes], [s[1] for s in image_shapes]
        hs, ws = [H // self.model.latent_downsample for H in Hs], [W // self.model.latent_downsample for W in Ws]
        
        grouped_latents = {}
        for idx, (latent, h, w) in enumerate(zip(latents, hs, ws)):
            key = (h, w)
            if key not in grouped_latents:
                grouped_latents[key] = []
            grouped_latents[key].append((idx, latent))
        
        results = [None] * len(latents)
        
        for (h, w), items in grouped_latents.items():
            indices = [idx for idx, _ in items]
            batch_latents = [latent for _, latent in items]
            
            batch_tensor = []
            for latent in batch_latents:
                tensor = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
                tensor = torch.einsum("nhwpqc->nchpwq", tensor)
                tensor = tensor.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
                batch_tensor.append(tensor)
            
            batch_tensor = torch.cat(batch_tensor, dim=0)
            batch_tensor = batch_tensor.to(
                device=self.vae_device,
                dtype=self.vae_dtype,
                non_blocking=True,
            )
            
            batch_images = self.vae_model.decode(batch_tensor)
            batch_images = (batch_images * 0.5 + 0.5).clamp(0, 1)
            batch_images = batch_images.permute(0, 2, 3, 1) * 255
            batch_images = batch_images.to(torch.uint8).cpu()
            
            for i, idx in enumerate(indices):
                img_array = batch_images[i].numpy()
                results[idx] = Image.fromarray(img_array)
        
        return results

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        past_key_values = gen_context['past_key_values']
        kv_lens = list(gen_context['kv_lens'])
        ropes = list(gen_context['ropes'])

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        # generate_text mutated past_key_values in-place by `n_added` tokens per
        # sample; write back to gen_context so kv_lens / ropes stay consistent
        # with the cache. Without this, the next call (gen_image or
        # update_context_*) reads stale lens and produces shape mismatches.
        n_added = int(unpacked_latent.shape[0])
        gen_context['kv_lens'] = [k + n_added for k in kv_lens]
        gen_context['ropes'] = [r + n_added for r in ropes]
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        noise_seed=None,
        use_cfg_parallel=False,
        max_rounds:int=3,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                rounds = 0
                while rounds < max_rounds:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    output_list.append(gen_text)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    
                    if "<image_start>" in gen_text:
                        img = self.gen_image(
                            image_shapes, 
                            gen_context, 
                            cfg_text_precontext=cfg_text_context, 
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale, 
                            cfg_img_scale=cfg_img_scale, 
                            cfg_interval=cfg_interval, 
                            timestep_shift=timestep_shift, 
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            noise_seed=noise_seed,
                        )
                        output_list.append(img)

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                        rounds += 1
                    else:
                        break

        return output_list

    @torch.no_grad()
    def force_interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_think_token_n=1000,
        max_answer_token_n=4096,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        noise_seed=None,
    ) -> List[Union[str, Image.Image]]:
        """Forced single-round interleaved baseline.

        Pipeline (fully scripted, no <image_start> probing):
          P0: GEN_THINK_SYSTEM_PROMPT  (mirrors interleave_inference understanding_output=False conventions)
          P1: ingest user input text+image
          P2: gen_text → think_text     (max_think_token_n)
          P3: gen_image (UNCONDITIONAL — this is the "force")
          P4: feed image back as vit-only context
          P5: gen_text → answer_text   (max_answer_token_n)

        Returns: [think_text, PIL.Image, answer_text] (always 3 items).
        """
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # P0: system prompt (sticks to VLM_THINK because answer is text)
            gen_context = self.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
            cfg_img_context = self.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

            # P1: ingest user inputs (mirrors interleave_inference understanding_output=True path)
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=True)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            # P2: think text — no <image_start> gating, take whatever the model produces.
            # gen_text now updates gen_context['kv_lens']/['ropes'] internally to
            # match the cache extension; no separate update_context_text needed
            # (calling it again would re-encode the text and double-extend cache).
            think_text = self.gen_text(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n,
            )
            output_list.append(think_text)

            # P3: force gen_image (THE actual "force")
            img = self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                noise_seed=noise_seed,
                enable_taylorseer=enable_taylorseer,
            )
            output_list.append(img)

            # P4: feed image back as vit-only context (model "sees" it for the answer)
            img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
            gen_context = self.update_context_image(img_input, gen_context, vae=True, vit=True)

            # P5: final answer text
            answer_text = self.gen_text(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_answer_token_n,
            )
            output_list.append(answer_text)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        input_list = None, 
        **kargs
    ) -> Dict[str, Any]:
    
        if input_list is None:
            output_dict = {'image': None, 'text': None}

            if image is None and text is None:
                print('Please provide at least one input: either an image or text.')
                return output_dict

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        return output_list
