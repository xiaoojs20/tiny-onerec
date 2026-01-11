import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq

from collections import Counter, defaultdict
import math
import numpy as np
import json


def _entropy_from_counter(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0
    H = 0.0
    for c in counter.values():
        p = c / total
        H -= p * math.log(p + 1e-12)
    ppl = math.exp(H)
    return H, ppl

def _cosine(a: np.ndarray, b: np.ndarray):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def compute_sid_metrics(codes: np.ndarray, num_emb_list, embs: np.ndarray | None = None,
                        pairs_per_code: int = 20, max_keep_per_code: int = 30):
    """
    codes: (N, L) int32
    num_emb_list: [K1, K2, ... KL]
    embs: (N, D) float32, optional for PAS
    """
    N, L = codes.shape
    metrics = {}

    # ---- full SID collision ----
    full = [tuple(row) for row in codes.tolist()]
    full_cnt = Counter(full)
    metrics["collision_full"] = (N - len(full_cnt)) / N
    metrics["unique_rate_full"] = len(full_cnt) / N
    bucket_sizes = np.array(list(full_cnt.values()), dtype=np.int32)
    metrics["max_bucket_full"] = int(bucket_sizes.max())
    metrics["p95_bucket_full"] = float(np.percentile(bucket_sizes, 95))

    # ---- prefix collisions & bucket sizes ----
    for d in range(1, L + 1):
        pref = [tuple(row[:d]) for row in codes[:, :d].tolist()]
        pc = Counter(pref)
        sizes = np.array(list(pc.values()), dtype=np.int32)
        metrics[f"collision_prefix@{d}"] = (N - len(pc)) / N
        metrics[f"max_bucket_prefix@{d}"] = int(sizes.max())

    # ---- per-layer utilization & entropy ----
    for l in range(L):
        K = int(num_emb_list[l])
        lc = Counter(codes[:, l].tolist())
        cur = len(lc) / K
        H, ppl = _entropy_from_counter(lc)
        metrics[f"CUR@L{l+1}"] = cur
        metrics[f"entropy@L{l+1}"] = H
        metrics[f"perplexity@L{l+1}"] = ppl
        metrics[f"top1_share@L{l+1}"] = max(lc.values()) / N

    # ---- PAS (sampled) ----
    if embs is not None:
        # keep a bounded number of examples per code to control memory / compute
        buckets = defaultdict(list)
        for i, c in enumerate(full):
            if len(buckets[c]) < max_keep_per_code:
                buckets[c].append(i)

        sims = []
        rng = np.random.default_rng(2024)
        for idxs in buckets.values():
            m = len(idxs)
            if m < 2:
                continue
            # sample a few pairs
            num_pairs = min(pairs_per_code, m * (m - 1) // 2)
            for _ in range(num_pairs):
                i, j = rng.choice(idxs, size=2, replace=False)
                sims.append(_cosine(embs[i], embs[j]))

        metrics["PAS_full"] = float(np.mean(sims)) if sims else float("nan")

    return metrics

def _format_metrics_for_log(metrics: dict, num_emb_list) -> str:
    # 先打印全局
    keys = ["collision_full", "max_bucket_full", "p95_bucket_full", "PAS_full"]
    parts = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            parts.append(f"{k}={v:.6f}" if isinstance(v, (float, int)) else f"{k}={v}")

    # 再按层展开（按 num_emb_list 期望层数，或按 metrics["_L"] 实际层数）
    L = int(metrics.get("_L", len(num_emb_list)))
    for i in range(1, L + 1):
        for k in (f"CUR@L{i}", f"entropy@L{i}", f"perplexity@L{i}", f"top1_share@L{i}"):
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.6f}")

    return ", ".join(parts)


class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num   # data_num: steps(number of batches) per epoch
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit                   # 最多保留多少个 checkpoint
        self.best_save_heap = []                            # 一个小根堆，用来保存“碰撞率最好”的若干 ckpt   
        self.newest_save_queue = []                         # 一个 FIFO 队列，保存最近 save_limit 个 ckpt
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    # @torch.no_grad()
    # def _valid_epoch(self, valid_data):

    #     self.model.eval()

    #     all_codes = []
    #     all_embs = []  # 可选：不想算 PAS 就删掉这行及下面 append

    #     iter_data =tqdm(
    #             valid_data,
    #             total=len(valid_data),
    #             ncols=100,
    #             desc=set_color(f"Evaluate   ", "pink"),
    #         )

    #     indices_set = set()
    #     num_sample = 0
    #     for batch_idx, data in enumerate(iter_data):
    #         num_sample += len(data)
    #         # data: (B, D)
    #         all_embs.append(data.cpu().numpy().astype(np.float32))

    #         data = data.to(self.device)
    #         indices = self.model.get_indices(data)
    #         indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
    #         for index in indices:
    #             code = "-".join([str(int(_)) for _ in index])
    #             indices_set.add(code)

    #         all_codes.append(indices.astype(np.int32))

    #     codes = np.concatenate(all_codes, axis=0)
    #     embs  = np.concatenate(all_embs, axis=0)


    #     # 极端情况下，collision_rate = (n - 1) / n ~= 1
    #     collision_rate = (num_sample - len(list(indices_set)))/num_sample

    #     metrics = compute_sid_metrics(
    #         codes=codes,
    #         num_emb_list=self.args.num_emb_list,
    #         embs=embs,              # 不算 PAS 就传 None
    #         pairs_per_code=20
    #     )
    #     return metrics
    #     # return collision_rate


    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        all_codes, all_embs = [], []

        iter_data = tqdm(
            valid_data, total=len(valid_data), ncols=100,
            desc=set_color("Evaluate", "pink"),
        )

        for _, data in enumerate(iter_data):
            # data 在 DataLoader 出来默认是 CPU tensor
            all_embs.append(data.numpy().astype(np.float32))

            indices = self.model.get_indices(data.to(self.device))
            indices_np = _to_numpy(indices)

            # 统一成 (B, L)
            indices_np = indices_np.reshape(-1, indices_np.shape[-1]).astype(np.int32)
            all_codes.append(indices_np)

        codes = np.concatenate(all_codes, axis=0)
        embs = np.concatenate(all_embs, axis=0)

        metrics = compute_sid_metrics(
            codes=codes,
            num_emb_list=self.args.num_emb_list,
            embs=embs,
            pairs_per_code=20
        )
        # 可选：把实际层数也塞回去，方便 fit 自动打印
        metrics["_L"] = int(codes.shape[1])
        metrics["_N"] = int(codes.shape[0])
        return metrics
    

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    # def fit(self, data):

    #     cur_eval_step = 0

    #     for epoch_idx in range(self.epochs):
    #         # train
    #         training_start_time = time()
    #         train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
    #         training_end_time = time()
    #         train_loss_output = self._generate_train_loss_output(
    #             epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
    #         )
    #         self.logger.info(train_loss_output)


    #         # eval
    #         if (epoch_idx + 1) % self.eval_step == 0:
    #             valid_start_time = time()
    #             collision_rate = self._valid_epoch(data)

    #             if train_loss < self.best_loss:
    #                 self.best_loss = train_loss
    #                 self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

    #             if collision_rate < self.best_collision_rate:
    #                 self.best_collision_rate = collision_rate
    #                 cur_eval_step = 0
    #                 self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
    #                                       ckpt_file=self.best_collision_ckpt)
    #             else:
    #                 cur_eval_step += 1


    #             valid_end_time = time()
    #             valid_score_output = (
    #                 set_color("epoch %d evaluating", "green")
    #                 + " ["
    #                 + set_color("time", "blue")
    #                 + ": %.2fs, "
    #                 + set_color("collision_rate", "blue")
    #                 + ": %f]"
    #             ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

    #             self.logger.info(valid_score_output)
    #             ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
    #             now_save = (-collision_rate, ckpt_path)
    #             if len(self.newest_save_queue) < self.save_limit:
    #                 self.newest_save_queue.append(now_save)
    #                 heapq.heappush(self.best_save_heap, now_save)
    #             else:
    #                 old_save = self.newest_save_queue.pop(0)
    #                 self.newest_save_queue.append(now_save)
    #                 if collision_rate < -self.best_save_heap[0][0]:
    #                     bad_save = heapq.heappop(self.best_save_heap)
    #                     heapq.heappush(self.best_save_heap, now_save)

    #                     if bad_save not in self.newest_save_queue:
    #                         delete_file(bad_save[1])

    #                 if old_save not in self.best_save_heap:
    #                     delete_file(old_save[1])



    #     return self.best_loss, self.best_collision_rate



    
    def fit(self, data):
        # 追加写：每次 eval 一行，方便后处理画曲线
        metrics_path = os.path.join(self.ckpt_dir, "metrics.jsonl")

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # ---- train ----
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()

            self.logger.info(
                self._generate_train_loss_output(
                    epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
                )
            )

            # ---- eval ----
            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            valid_start_time = time()
            valid_out = self._valid_epoch(data)  # 现在可能是 dict，也可能是 float（兼容）

            if isinstance(valid_out, dict):
                metrics = valid_out
                collision_rate = float(metrics.get("collision_full", metrics.get("collision_rate", 1.0)))
            else:
                collision_rate = float(valid_out)
                metrics = {"collision_full": collision_rate}

            # ---- save best loss ----
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

            # ---- save best collision ----
            if collision_rate < self.best_collision_rate:
                self.best_collision_rate = collision_rate
                cur_eval_step = 0
                self._save_checkpoint(epoch_idx, collision_rate=collision_rate, ckpt_file=self.best_collision_ckpt)
            else:
                cur_eval_step += 1

            valid_end_time = time()

            # ---- logging: print more metrics ----
            extra = _format_metrics_for_log(metrics, self.args.num_emb_list)

            valid_score_output = (
                set_color(f"epoch {epoch_idx} evaluating", "green")
                + " ["
                + set_color("time", "blue")
                + f": {valid_end_time - valid_start_time:.2f}s, "
                + extra
                + "]"
            )
            self.logger.info(valid_score_output)


            # ---- append metrics to jsonl (safe for append) ----
            record = {
                "epoch": epoch_idx,
                "train_loss": float(train_loss),
                "train_recon_loss": float(train_recon_loss),
                "collision_full": float(collision_rate),
                "metrics": metrics,
                "ts": get_local_time(),
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # ---- save regular checkpoint + keep only some ----
            ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
            now_save = (-collision_rate, ckpt_path)

            if len(self.newest_save_queue) < self.save_limit:
                self.newest_save_queue.append(now_save)
                heapq.heappush(self.best_save_heap, now_save)
            else:
                old_save = self.newest_save_queue.pop(0)
                self.newest_save_queue.append(now_save)

                if collision_rate < -self.best_save_heap[0][0]:
                    bad_save = heapq.heappop(self.best_save_heap)
                    heapq.heappush(self.best_save_heap, now_save)
                    if bad_save not in self.newest_save_queue:
                        delete_file(bad_save[1])

                if old_save not in self.best_save_heap:
                    delete_file(old_save[1])

        self.logger.info(f"[METRICS] saved to: {metrics_path}")
        return self.best_loss, self.best_collision_rate




