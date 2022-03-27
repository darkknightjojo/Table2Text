"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""
import copy

from torch.nn.utils.rnn import pad_sequence

import torch
import traceback

from core.metrics import RLLossCompute
from core.modules.sparse_activations import LogSparsemax
from core.utils.misc import get_model_device
from core.utils.logging import logger
from core.models.model_switch import SwitchModel
import core.utils


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`core.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`core.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`core.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = core.utils.loss.build_loss_compute(model, tgt_field, opt)
    lm_train_loss = core.utils.loss.build_loss_compute(model, tgt_field, opt, lm=True)
    valid_loss = core.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = core.utils.EarlyStopping(
        opt.early_stopping, scorers=core.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    tabbie_embeddings = opt.tabbie_embeddings if opt.tabbie_embeddings else None
    weights_file = opt.weights_file if opt.weights_file else None

    report_manager = core.utils.build_report_manager(opt, gpu_rank)
    return Trainer(model, train_loss, valid_loss, optim, trunc_size,
                   shard_size, norm_method,
                   accum_count, accum_steps,
                   n_gpu, gpu_rank,
                   gpu_verbose_level, report_manager,
                   with_align=True if opt.lambda_align > 0 else False,
                   model_saver=model_saver if gpu_rank == 0 else None,
                   average_decay=average_decay,
                   average_every=average_every,
                   model_dtype=opt.model_dtype,
                   earlystopper=earlystopper,
                   dropout=dropout,
                   dropout_steps=dropout_steps,
                   tabbie_embeddings=tabbie_embeddings,
                   weights_file=weights_file,
                   lambda_lm=opt.lambda_lm,
                   train_lm=opt.train_lm,
                   nmt_lm_loss=opt.nmt_lm_loss,
                   add_nmt_lm_loss=opt.add_nmt_lm_loss,
                   add_nmt_lm_loss_fn=opt.add_nmt_lm_loss_fn,
                   lambda_add_loss=opt.lambda_add_loss,
                   fixed_lm=opt.fixed_lm,
                   report_lm=opt.report_lm,
                   lm_train_loss=lm_train_loss
                   )


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`core.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`core.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`core.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`core.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`core.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`core.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 fixed_lm=False,
                 nmt_lm_loss=-1.0,
                 add_nmt_lm_loss=False,
                 train_lm=False,
                 add_nmt_lm_loss_fn=None,
                 lambda_add_loss=1.0,
                 tabbie_embeddings=None,
                 weights_file=None,
                 lambda_lm=1e-2,
                 report_lm=False,
                 lm_train_loss=None
                 ):
        # Basic attributes.
        self.report_lm = report_lm
        self.lambda_lm = lambda_lm
        self.lambda_add_loss = lambda_add_loss
        self.add_nmt_lm_loss_fn = add_nmt_lm_loss_fn
        self.train_lm = train_lm
        self.add_nmt_lm_loss = add_nmt_lm_loss
        self.nmt_lm_loss = nmt_lm_loss
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.fixed_lm = fixed_lm
        self.loss_gen = None
        self.lm_embeddings = None
        self.lm_train_loss = lm_train_loss

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        self._dev = get_model_device(self.model)

        # Set model in training mode.
        self.model.train()

        # Load tabbie_embedding in case of pretrain_base rnn training
        if tabbie_embeddings:
            # 每个文件中存储的tensor的数量
            self.table_embeddings_count = 0
            # 当前加载的文件的编号
            self.table_embeddings_rank = 0
            # 文件保存路径
            self.table_embeddings_path = tabbie_embeddings
            # 加载进内存的张量
            self.table_embeddings = self.load_table_embeddings_file(tabbie_embeddings, 0)
            self.table_embeddings_count = len(self.table_embeddings)
        else:
            self.table_embeddings = None
            # Load weights in case of multi-branches rnn training

        if weights_file:
            self.decoder_rnn_weights = self.load_weights_file(weights_file)
        else:
            self.decoder_rnn_weights = None
    @staticmethod
    def load_weights_file(path):
        with open(path, mode="r", encoding="utf8") as f:
            return [
                torch.Tensor([list(map(float, s.split(':'))) for s in line.split()])
                for line in f
            ]

    @staticmethod
    def load_table_embeddings_file(path, rank):
        path = path + str(rank) + ".pt"
        return torch.load(path)

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = core.utils.Statistics()
        report_stats = core.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        if self.fixed_lm:
            self.loss_gen = copy.deepcopy(
                self.model.generator[0] if isinstance(self.model.generator[-1], LogSparsemax) else self.model.generator)
            self.lm_embeddings = copy.deepcopy(self.model.decoder.embeddings)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(core.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats, report_stats, self.train_lm, loss_gen=self.loss_gen,
                lm_embeddings=self.lm_embeddings)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = core.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths,
                                             with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats, report_stats, train_lm, loss_gen=None,
                               lm_embeddings=None):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                # 加载embedding或者权重文件
                kwargs = dict()
                if self.table_embeddings:
                    embeddings = []
                    for b in range(batch.batch_size):
                        # 计算文件编号
                        quotient = (batch.indices[b] // self.table_embeddings_count).item()
                        remainder = (batch.indices[b] % self.table_embeddings_count).item()
                        # 如果该文件已经加载，直接读取
                        if quotient == self.table_embeddings_rank:
                            embeddings.append(self.table_embeddings[remainder])
                        else:
                            # 释放显存
                            self.table_embeddings = None
                            torch.cuda.empty_cache()
                            self.table_embeddings = self.load_table_embeddings_file(self.table_embeddings_path,
                                                                                    quotient)
                            self.table_embeddings_rank = quotient
                            embeddings.append(self.table_embeddings[remainder])

                    kwargs['table_embeddings'] = embeddings

                if self.decoder_rnn_weights:
                    kwargs['dec_weights'] = pad_sequence([
                        torch.Tensor(self.decoder_rnn_weights[batch.indices[b].item()])
                        for b in range(batch.batch_size)
                    ], batch_first=False).to(self._dev)

                out = self.model(src, tgt, src_lengths, bptt=bptt, with_align=self.with_align, **kwargs)
                if len(out) == 3:
                    outputs, attns, lm_outputs = out
                else:
                    outputs, attns = out
                    lm_outputs = None
                del out
                bptt = True

                # 计算语言模型损失
                loss_add = None
                delta_weight = None
                if (self.nmt_lm_loss != -1.0 or self.add_nmt_lm_loss) and lm_outputs is not None:
                    # generator is nmt's generator, is changing during training
                    if loss_gen is None:  # not fixed lm model
                        generator = self.model.lm_generator[0] if isinstance(self.model.lm_generator[-1],
                                                                                        LogSparsemax) else self.model.lm_generator
                        loss_gen = generator

                    tgt_0 = tgt[1:]  # to compute language model pred_prob
                    tgt_in = tgt[:-1]  # the actual input of decoder
                    tgt_in = tgt_in[:, :, 0].transpose(0, 1)  # (batch_size, tgt_len)
                    pad_idx = 1
                    eos_idx = 3
                    tgt_pad_mask = tgt_in.data.eq(pad_idx)  # (batch_size, tgt_len)
                    tgt_eos_mask = tgt_in.data.eq(eos_idx)
                    mask = torch.gt(tgt_pad_mask + tgt_eos_mask, 0)

                    # lm preds
                    if self.train_lm:
                        lm_preds = loss_gen(lm_outputs)
                    else:  # not training lm model, so need detach
                        lm_outs = lm_outputs.detach()
                        lm_preds = loss_gen(lm_outs)  # （tgt_len, batch_size, vocab_size)
                        lm_preds = lm_preds.detach()
                    lm_preds = torch.gather(lm_preds, 2, tgt_0.long())  # (tgt_len, batch_size, 1)
                    lm_preds = lm_preds.squeeze(2).transpose(0, 1).contiguous()  # batch_size, len
                    lm_preds = torch.exp(lm_preds)

                    # nmt preds
                    nmt_preds = generator(outputs)  # （tgt_len, batch_size, vocab_size)
                    nmt_preds = torch.gather(nmt_preds, 2, tgt_0.long())  # (tgt_len, batch_size, 1)
                    nmt_preds = nmt_preds.squeeze(2).transpose(0, 1).contiguous()
                    nmt_preds = torch.exp(nmt_preds)

                    if self.add_nmt_lm_loss:
                        if self.add_nmt_lm_loss_fn == 'linear':  # linear-linear
                            nmt_lm = (1 - nmt_preds).mul(1 - nmt_preds + lm_preds) / 2.
                            # (1 - nmt) * (1 - nmt+lm)/2 , less is better
                        elif self.add_nmt_lm_loss_fn == 'x3':  # cube
                            nmt_lm = (1 - nmt_preds).mul(1 - (nmt_preds - lm_preds) ** 3) / 2
                            # (1-nmt)* (1-delta^3)/2
                        elif self.add_nmt_lm_loss_fn == 'x5':  # quintic
                            nmt_lm = (1 - nmt_preds).mul(1 - (nmt_preds - lm_preds) ** 5) / 2
                            # (1-nmt)* (1-delta^5)/2
                        elif self.add_nmt_lm_loss_fn == 'sigmoid_5':
                            k = 1. / 5.
                            nmt_lm = (1 - nmt_preds).mul(1. / (1 + torch.exp((nmt_preds - lm_preds) / k)))
                            # (1-nmt)* (1 + exp(delta/k))^(-1)
                        elif self.add_nmt_lm_loss_fn == 'sigmoid_10':
                            k = 1. / 10.
                            nmt_lm = (1 - nmt_preds).mul(1. / (1 + torch.exp((nmt_preds - lm_preds) / k)))
                        elif self.add_nmt_lm_loss_fn == 'log_10':
                            # y5 = np.log((0.5-x/2 + 1e-8)/(0.5+x/2 + 1e-8)) / k + 0.5
                            k = 10.
                            nmt_lm = (1 - nmt_preds).mul(torch.log((0.5 - nmt_preds / 2 + lm_preds / 2. + 1e-8) / (
                                    0.5 + nmt_preds / 2 - lm_preds / 2. + 1e-8)) / k + 0.5)
                        elif self.add_nmt_lm_loss_fn == 'log_5':
                            k = 5.
                            nmt_lm = (1 - nmt_preds).mul(torch.log((0.5 - nmt_preds / 2 + lm_preds / 2. + 1e-8) / (
                                    0.5 + nmt_preds / 2 - lm_preds / 2. + 1e-8)) / k + 0.5)
                        elif self.add_nmt_lm_loss_fn == 'log_20':
                            k = 20.
                            nmt_lm = (1 - nmt_preds).mul(torch.log((0.5 - nmt_preds / 2 + lm_preds / 2. + 1e-8) / (
                                    0.5 + nmt_preds / 2 - lm_preds / 2. + 1e-8)) / k + 0.5)

                        if nmt_lm.shape[1] > 2:
                            mask[:, :2] = True
                            # ce + lambda*(1-p(NMT))g(delta)   (token level)
                            loss_add = self.lambda_add_loss * nmt_lm[~mask].sum()

                            loss_add.div(float(normalization)).backward(retain_graph=True)
                    else:  # hard margin loss, if delta > threshold, loss = 0
                        nmt_lm = self.nmt_lm_loss - (nmt_preds - lm_preds)  # batch, len
                        if nmt_lm.shape[1] > 2:
                            nmt_lm[nmt_lm < 0] = 0.
                            mask[:, :2] = True  # don't account y1, y2
                            loss_add = self.lambda_add_loss * nmt_lm[~mask].sum()
                            loss_add.div(float(normalization)).backward(retain_graph=True)
                    print('loss add: ', loss_add.div(float(normalization)))

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size
                    )

                    lm_loss = None
                    if lm_outputs is not None and train_lm:
                        lm_loss, _ = self.lm_train_loss(
                            batch,
                            lm_outputs,
                            attns,
                            normalization=normalization,
                            shard_size=self.shard_size,
                            trunc_start=j,
                            trunc_size=trunc_size,
                            lm_output=True,
                            lm_lambda=self.lambda_lm)

                    if loss is not None:
                        # print('nmt loss:', loss)
                        if lm_loss is not None:
                            loss += lm_loss
                            # print('lm loss:', lm_loss)
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        core.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

                # if isinstance(self.model, SwitchModel):
                #     self.model(tgt, src, lengths=src_lengths, bptt=bptt, reverse=True, with_align=self.with_align, **kwargs)

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                core.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:core.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return core.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `core.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `core.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)


