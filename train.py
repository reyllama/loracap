import os
import sys
import argparse
import json
import types
from PIL import Image
import io
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dataset import cc30k
from func import forward_seq, forward_dec, forward_seq_dec, CoCa
import matplotlib.pyplot as plt
from util import cosine_scheduler

class Trainer:
    def __init__(self, args=None, model_config=None):
        self.args = args

        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using ", self.device)

        # model
        image_encoder, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        image_encoder.eval()
        image_encoder = image_encoder.visual
        image_encoder.forward = types.MethodType(forward_seq, image_encoder)
        self.image_encoder = CoCa(image_encoder).to(self.device)

        for name, param in self.image_encoder.named_parameters():
            if not ("img_queries" in name or "img_attn_pool" in name):
                param.requires_grad = False

        self.image_encoder.main_input_name = 'input_ids'

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model.decoder.forward = types.MethodType(forward_dec, self.model.decoder)

        # LoRA on FLAN-T5
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q", "v", "wi"]
        )

        self.model.decoder = get_peft_model(self.model.decoder, config)
        self.model.decoder.print_trainable_parameters()

        self.model.encoder = self.image_encoder
        self.model.to(self.device)

        # dataset
        self.dataset = cc30k(dir=args.dataset, transform=preprocess)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4, batch_size=args.batch)
        self.val_dataset = cc30k(dir=args.val_dataset, transform=preprocess)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, num_workers=4, batch_size=args.batch)

        # optimizer
        params_to_optimize = list(self.model.decoder.parameters()) + list(self.model.encoder.parameters())
        self.optimizer = optim.AdamW(params_to_optimize, lr=args.base_lr)
        self.lr_scheduler = cosine_scheduler(args.base_lr, args.final_lr, args.epoch, len(self.dataloader), args.warmup_epoch, args.initial_lr)

    def train(self):

        step = 0
        train_loss = 0
        losses = {'train_loss': [], 'val_loss': []}
        os.makedirs(os.path.join(self.args.output_path, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_path, "curves"), exist_ok=True)

        for ep in range(self.args.epoch):
            for data in self.dataloader:
                image, caption = data['image'], data['caption']
                for k, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.lr_scheduler[step]
                step += 1
                self.optimizer.zero_grad()
                image = image.to(self.device)
                caption = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
                loss = self.model(input_ids=image, labels=caption.input_ids).loss
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if step % self.args.save_every == 0:
                    val_loss = self.evaluate()
                    log_stats = {'step': step,
                                 'running_loss': f"{(train_loss/self.args.save_every):.4f}",
                                 'running_val_loss': f"{val_loss:.4f}"}
                    losses['train_loss'].append(train_loss/self.args.save_every)
                    losses['val_loss'].append(val_loss)
                    train_loss = 0

                    with open(os.path.join(self.args.output_path, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    print(f"step: {step}, loss: {loss.item():.4f}")

            self._save(self.model, f"{ep+1}.pt")
            self._plot_curves(losses, ['train', 'val'], ep+1)
            self._plot_figure(ep + 1)

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        for i, data in enumerate(self.val_dataloader):
            image, caption = data['image'], data['caption']
            image = image.to(self.device)
            caption = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            loss = self.model(input_ids=image, labels=caption.input_ids).loss
            val_loss += loss.item()
        self.model.train()
        return val_loss / len(self.val_dataloader)

    def _save(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.args.output_path, name))

    def _plot_curves(self, stats, names, ep):
        assert len(stats['train_loss']) == len(stats['val_loss']), "train and val loss should have same length"

        plt.figure(figsize=(8, 8))
        for name in names:
            iterations = range(0, len(stats[name + '_loss']) * self.args.save_every, self.args.save_every)
            plt.plot(iterations, stats[name + '_loss'], label=name)

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(loc='upper right')
        plt.savefig(f'{self.args.output_path}/curves/{ep}_loss_curves.png', dpi=300)

    @torch.no_grad()
    def _plot_figure(self, ep):
        tik = 0
        for data in self.val_dataloader:
            images, captions = data['image'].to(self.device), data['caption']
            preds = self.model.generate(images)
            gen_captions = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            for image, caption in zip(images, gen_captions):
                plt.figure(figsize=(8, 8))
                plt.imshow(image.cpu().permute(1, 2, 0))
                plt.title(' '.join(caption), fontdict={'fontsize': 8})
                plt.axis('off')
                plt.savefig(f'{self.args.output_path}/samples/{ep}_{tik}.png', dpi=300)
                tik += 1
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eval', action="store_true", help="evaluation only")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='/home/user/Data/conceptualcaps/cc3m_300k')
    parser.add_argument('--val_dataset', type=str, default='/home/user/Data/conceptualcaps/cc3m_val')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--warmup_epoch', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--initial_lr', type=float, default=1e-5)
    parser.add_argument('--num_img_queries', type=int, default=256)
    parser.add_argument('--output_path', type=str, default='experiments/x3')

    args = parser.parse_args()
    model_config = {'num_img_queries': args.num_img_queries}
    trainer = Trainer(args=args, model_config=model_config)

    if args.eval:
        score = trainer.evaluate(ep=-1)
        print(f"[Epoch {args.ckpt}] Validation Accuracy {score:.4f}")
        sys.exit(0)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    trainer.train()