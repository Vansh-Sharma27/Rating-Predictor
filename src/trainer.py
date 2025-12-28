import os, json
from datetime import datetime
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from .metrics import RatingMetrics

class Trainer:
    def __init__(self, model, tokenizer, loss_fn, config, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn.to(device)
        self.cfg = config
        self.device = device

        tcfg = config["training"]
        self.grad_accum = tcfg["gradient_accumulation_steps"]
        self.max_steps = tcfg["max_steps"]
        self.eval_steps = tcfg["eval_steps"]
        self.save_steps = tcfg["save_steps"]
        self.log_steps = tcfg["logging_steps"]
        self.max_grad_norm = tcfg["max_grad_norm"]
        self.fp16 = tcfg["fp16"]

        self.patience = tcfg["early_stopping_patience"]
        self.threshold = tcfg["early_stopping_threshold"]

        self.ckpt_dir = config["paths"]["checkpoint_dir"]
        self.res_dir = config["paths"]["results_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

        self.global_step = 0
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.patience_counter = 0
        self.training_log = []

        self.scaler = GradScaler("cuda") if self.fp16 else None

    def _setup_optim(self):
        tcfg = self.cfg["training"]
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": tcfg["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"], betas=(0.9, 0.999), eps=1e-8)

        warmup = int(self.max_steps * tcfg["warmup_ratio"])
        self.scheduler = get_scheduler(
            name=tcfg["lr_scheduler_type"],
            optimizer=self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=self.max_steps,
        )
        print(f"Optimizer: {self.max_steps} steps, {warmup} warmup")

    def collate_fn(self, batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts, padding="longest", truncation=True, max_length=self.cfg["tokenizer"]["max_length"], return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def _train_micro_step(self, batch) -> float:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.fp16:
            with autocast("cuda"):
                out = self.model(input_ids, attention_mask)
                loss = self.loss_fn(out["logits"], out["regression"], labels) / self.grad_accum
            self.scaler.scale(loss).backward()
            return float(loss.item() * self.grad_accum)
        else:
            out = self.model(input_ids, attention_mask)
            loss = self.loss_fn(out["logits"], out["regression"], labels) / self.grad_accum
            loss.backward()
            return float(loss.item() * self.grad_accum)

    def _optim_step(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, n = 0.0, 0

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.fp16:
                with autocast("cuda"):
                    out = self.model(input_ids, attention_mask)
                    loss = self.loss_fn(out["logits"], out["regression"], labels)
            else:
                out = self.model(input_ids, attention_mask)
                loss = self.loss_fn(out["logits"], out["regression"], labels)

            total_loss += float(loss.item())
            n += 1

            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        m = RatingMetrics.compute(np.array(all_labels), np.array(all_preds))
        m["loss"] = total_loss / max(n, 1)
        self.model.train()
        return m

    def _save_checkpoint(self, name: str, metrics=None):
        path = os.path.join(self.ckpt_dir, name)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_f1": self.best_f1,
            "best_acc": self.best_acc,
            "config": self.cfg,
            "metrics": metrics,
        }
        if self.fp16:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        print(f"  Saved: {name}")

    def _save_log(self):
        path = os.path.join(self.res_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(
                {"log": self.training_log, "best_f1": self.best_f1, "best_acc": self.best_acc,
                 "total_steps": self.global_step, "timestamp": datetime.now().isoformat()},
                f, indent=2
            )

    def train(self, train_loader, val_loader):
        self._setup_optim()
        self.model.train()

        it = iter(train_loader)
        running, accum = 0.0, 0
        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            running += self._train_micro_step(batch)
            accum += 1

            if accum >= self.grad_accum:
                self._optim_step()
                self.global_step += 1
                accum = 0
                pbar.update(1)

                if self.global_step % self.log_steps == 0:
                    avg = running / self.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr:.2e}"})
                    self.training_log.append({"step": self.global_step, "loss": avg, "lr": lr})
                    running = 0.0

                if self.global_step % self.eval_steps == 0:
                    print(f"\nStep {self.global_step}: Evaluation")
                    m = self.evaluate(val_loader)
                    print(f"  Loss: {m['loss']:.4f}")
                    print(f"  Accuracy: {m['accuracy']:.4f}")
                    print(f"  F1 Macro: {m['f1_macro']:.4f}")
                    print(f"  Off-by-one: {m['off_by_one_acc']:.4f}")

                    if m["f1_macro"] > self.best_f1 + self.threshold:
                        self.best_f1 = m["f1_macro"]
                        self.best_acc = m["accuracy"]
                        self.patience_counter = 0
                        self._save_checkpoint("best_model.pt", m)
                        print(f"  ✓ New best! F1={self.best_f1:.4f}, Acc={self.best_acc:.4f}")
                    else:
                        self.patience_counter += 1
                        print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")

                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at step {self.global_step}")
                        break

                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"checkpoint_{self.global_step}.pt")

        pbar.close()
        self._save_checkpoint("final_model.pt")
        self._save_log()
        print(f"\nTraining complete. Best F1: {self.best_f1:.4f}, Best Acc: {self.best_acc:.4f}")
        return self.best_f1
