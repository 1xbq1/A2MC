import torch
import torch.nn as nn

from .scd_encoder import PretrainingEncoder


# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')

class Adversary_Negatives(nn.Module):
    def __init__(self,bank_size,dim,multi_crop=0):
        super(Adversary_Negatives, self).__init__()
        self.multi_crop = multi_crop
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q, init_mem=False):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        if self.multi_crop and not init_mem:
            logit_list = []
            for q_item in q:
                logit = torch.einsum('nc,ck->nk', [q_item, memory_bank])
                logit_list.append(logit)
            return memory_bank, self.W, logit_list
        else:
            logit=torch.einsum('nc,ck->nk', [q, memory_bank])
            return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

class SCD_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SCD_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        self.attack_fc = nn.Linear(2 * args_encoder['hidden_size'], args_encoder['num_class'])
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)
        weights_init(self.attack_fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # domain level queues
        # temporal domain queue
        self.t_queue = Adversary_Negatives(K, dim)
        # self.register_buffer("t_queue", torch.randn(dim, K))
        # self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        # self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.s_queue = Adversary_Negatives(K, dim)
        # self.register_buffer("s_queue", torch.randn(dim, K))
        # self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        # self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.i_queue = Adversary_Negatives(K, dim)
        # self.register_buffer("i_queue", torch.randn(dim, K))
        # self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        # self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, t_keys, s_keys, i_keys):
    #     N, C = t_keys.shape

    #     assert self.K % N == 0  # for simplicity

    #     t_ptr = int(self.t_queue_ptr)
    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
    #     t_ptr = (t_ptr + N) % self.K  # move pointer
    #     self.t_queue_ptr[0] = t_ptr

    #     s_ptr = int(self.s_queue_ptr)
    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
    #     s_ptr = (s_ptr + N) % self.K  # move pointer
    #     self.s_queue_ptr[0] = s_ptr

    #     i_ptr = int(self.i_queue_ptr)
    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
    #     i_ptr = (i_ptr + N) % self.K  # move pointer
    #     self.i_queue_ptr[0] = i_ptr

    def forward(self, qa_input, k_input, init_me=False, epoch=-1):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """

        # attack
        if k_input is None:
            vt, vs = self.encoder_q(qa_input, attack=True)
            vi = torch.cat([vt, vs], dim=1)
            qi = self.attack_fc(vi)
            return qi
        
        if init_me:
            qt, qs, qi = self.encoder_q(qa_input)  # queries: NxC
            qt = nn.functional.normalize(qt, dim=1)
            qs = nn.functional.normalize(qs, dim=1)
            qi = nn.functional.normalize(qi, dim=1)

            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                kt, ks, ki = self.encoder_k(k_input)  # keys: NxC
                kt = nn.functional.normalize(kt, dim=1)
                ks = nn.functional.normalize(ks, dim=1)
                ki = nn.functional.normalize(ki, dim=1)
            
            l_pos_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
            l_pos_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
            l_pos_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
            l_pos_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)
            _, _, l_neg_ti = self.i_queue(qt)
            _, _, l_neg_si = self.i_queue(qs)
            _, _, l_neg_it = self.t_queue(qi)
            _, _, l_neg_is = self.s_queue(qi)

            logits_ti = torch.cat([l_pos_ti, l_neg_ti], dim=1)
            logits_si = torch.cat([l_pos_si, l_neg_si], dim=1)
            logits_it = torch.cat([l_pos_it, l_neg_it], dim=1)
            logits_is = torch.cat([l_pos_is, l_neg_is], dim=1)
            logits_ti /= self.T
            logits_si /= self.T
            logits_it /= self.T
            logits_is /= self.T

            labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long).cuda()
            labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long).cuda()
            labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long).cuda()
            labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long).cuda()

            # fill the memory bank
            batch_size = kt.size(0)
            start_point = epoch * batch_size
            end_point = min((epoch + 1) * batch_size, self.K)
            self.i_queue.W.data[:, start_point:end_point] = ki[:end_point - start_point].T
            self.t_queue.W.data[:, start_point:end_point] = kt[:end_point - start_point].T
            self.s_queue.W.data[:, start_point:end_point] = ks[:end_point - start_point].T

            return logits_ti, logits_si, logits_it, logits_is, \
               labels_ti, labels_si, labels_it, labels_is,
    
        else:
            # compute temporal domain level, spatial domain level and instance level features
            #qt, qs, qi = self.encoder_q(q_input)  # queries: NxC
            #qt = nn.functional.normalize(qt, dim=1)
            #qs = nn.functional.normalize(qs, dim=1)
            #qi = nn.functional.normalize(qi, dim=1)

            qat, qas, qai = self.encoder_q(qa_input)  # queries: NxC
            qat = nn.functional.normalize(qat, dim=1)
            qas = nn.functional.normalize(qas, dim=1)
            qai = nn.functional.normalize(qai, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                kt, ks, ki = self.encoder_k(k_input)  # keys: NxC
                kt = nn.functional.normalize(kt, dim=1)
                ks = nn.functional.normalize(ks, dim=1)
                ki = nn.functional.normalize(ki, dim=1)

            # interactive loss
            #l_pos_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
            #l_pos_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
            #l_pos_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
            #l_pos_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)
            l_pos_ati = torch.einsum('nc,nc->n', [qat, ki]).unsqueeze(1)
            l_pos_asi = torch.einsum('nc,nc->n', [qas, ki]).unsqueeze(1)
            l_pos_ait = torch.einsum('nc,nc->n', [qai, kt]).unsqueeze(1)
            l_pos_ais = torch.einsum('nc,nc->n', [qai, ks]).unsqueeze(1)

            mix_r_pos = 0.2
            mix_r_neg = 0.1
            rep = self.K // qat.size(0)

            '''d_norm_ti, d_ti, l_neg_ti = self.i_queue(qt)
            mix_neg_ti = mix_r*(qt.repeat(rep,1)) + (1-mix_r)*(d_norm_ti.T)
            l_neg_mix_ti = torch.einsum('nc,ck->nk', [qt, mix_neg_ti.T])

            d_norm_si, d_si, l_neg_si = self.i_queue(qs)
            mix_neg_si = mix_r*(qs.repeat(rep,1)) + (1-mix_r)*(d_norm_si.T)
            l_neg_mix_si = torch.einsum('nc,ck->nk', [qs, mix_neg_si.T])

            d_norm_it, d_it, l_neg_it = self.t_queue(qi)
            mix_neg_it = mix_r*(qi.repeat(rep,1)) + (1-mix_r)*(d_norm_it.T)
            l_neg_mix_it = torch.einsum('nc,ck->nk', [qi, mix_neg_it.T])

            d_norm_is, d_is, l_neg_is = self.s_queue(qi)
            mix_neg_is = mix_r*(qi.repeat(rep,1)) + (1-mix_r)*(d_norm_is.T)
            l_neg_mix_is = torch.einsum('nc,ck->nk', [qi, mix_neg_is.T])'''

            d_norm_ati, d_ati, l_neg_ati = self.i_queue(qat)
            mix_neg_ati = mix_r_neg*(qat.repeat(rep,1)) + (1-mix_r_neg)*(d_norm_ati.T)
            l_neg_mix_ati = torch.einsum('nc,ck->nk', [qat, mix_neg_ati.T])
            mix_pos_ati = (1-mix_r_pos)*(qat.repeat(rep,1)) + mix_r_pos*(d_norm_ati.T)
            l_pos_mix_ati = torch.einsum('nc,ck->nk', [qat, mix_pos_ati.T])

            d_norm_asi, d_asi, l_neg_asi = self.i_queue(qas)
            mix_neg_asi = mix_r_neg*(qas.repeat(rep,1)) + (1-mix_r_neg)*(d_norm_asi.T)
            l_neg_mix_asi = torch.einsum('nc,ck->nk', [qas, mix_neg_asi.T])
            mix_pos_asi = (1-mix_r_pos)*(qas.repeat(rep,1)) + mix_r_pos*(d_norm_asi.T)
            l_pos_mix_asi = torch.einsum('nc,ck->nk', [qas, mix_pos_asi.T])

            d_norm_ait, d_ait, l_neg_ait = self.t_queue(qai)
            mix_neg_ait = mix_r_neg*(qai.repeat(rep,1)) + (1-mix_r_neg)*(d_norm_ait.T)
            l_neg_mix_ait = torch.einsum('nc,ck->nk', [qai, mix_neg_ait.T])
            mix_pos_ait = (1-mix_r_pos)*(qai.repeat(rep,1)) + mix_r_pos*(d_norm_ait.T)
            l_pos_mix_ait = torch.einsum('nc,ck->nk', [qai, mix_pos_ait.T])

            d_norm_ais, d_ais, l_neg_ais = self.s_queue(qai)
            mix_neg_ais = mix_r_neg*(qai.repeat(rep,1)) + (1-mix_r_neg)*(d_norm_ais.T)
            l_neg_mix_ais = torch.einsum('nc,ck->nk', [qai, mix_neg_ais.T])
            mix_pos_ais = (1-mix_r_pos)*(qai.repeat(rep,1)) + mix_r_pos*(d_norm_ais.T)
            l_pos_mix_ais = torch.einsum('nc,ck->nk', [qai, mix_pos_ais.T])

            '''logits_ti = torch.cat([l_pos_ti, l_neg_ti, l_neg_mix_ti], dim=1)
            logits_si = torch.cat([l_pos_si, l_neg_si, l_neg_mix_si], dim=1)
            logits_it = torch.cat([l_pos_it, l_neg_it, l_neg_mix_it], dim=1)
            logits_is = torch.cat([l_pos_is, l_neg_is, l_neg_mix_is], dim=1)'''
            # logits_ati = torch.cat([l_pos_ati, l_neg_ati, l_neg_mix_ati], dim=1)
            # logits_asi = torch.cat([l_pos_asi, l_neg_asi, l_neg_mix_asi], dim=1)
            # logits_ait = torch.cat([l_pos_ait, l_neg_ait, l_neg_mix_ait], dim=1)
            # logits_ais = torch.cat([l_pos_ais, l_neg_ais, l_neg_mix_ais], dim=1)
            logits_ati = torch.cat([l_pos_ati, l_neg_ati], dim=1)
            logits_asi = torch.cat([l_pos_asi, l_neg_asi], dim=1)
            logits_ait = torch.cat([l_pos_ait, l_neg_ait], dim=1)
            logits_ais = torch.cat([l_pos_ais, l_neg_ais], dim=1)

            identity_mat = torch.eye(l_pos_mix_ati.shape[0])
            mask_pos_mat = identity_mat.repeat(1, rep).cuda()
            
            l_pos_mix_ati /= self.T
            l_pos_mix_asi /= self.T
            l_pos_mix_ait /= self.T
            l_pos_mix_ais /= self.T

            loss_pos_ati = -1.0*(torch.log(torch.exp(l_pos_mix_ati))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_pos_asi = -1.0*(torch.log(torch.exp(l_pos_mix_asi))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_pos_ait = -1.0*(torch.log(torch.exp(l_pos_mix_ait))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_pos_ais = -1.0*(torch.log(torch.exp(l_pos_mix_ais))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)

            l_neg_mix_ati /= self.T
            l_neg_mix_asi /= self.T
            l_neg_mix_ait /= self.T
            l_neg_mix_ais /= self.T

            loss_neg_ati = 1.0*(torch.log(torch.exp(l_neg_mix_ati))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_neg_asi = 1.0*(torch.log(torch.exp(l_neg_mix_asi))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_neg_ait = 1.0*(torch.log(torch.exp(l_neg_mix_ait))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)
            loss_neg_ais = 1.0*(torch.log(torch.exp(l_neg_mix_ais))*mask_pos_mat).sum()/(mask_pos_mat.sum() + 1E-8)

            loss_mix = (loss_pos_ati + loss_pos_asi + loss_pos_ait + loss_pos_ais) / 4.0 + \
                       (loss_neg_ati + loss_neg_asi + loss_neg_ait + loss_neg_ais) / 4.0

            '''logits_ti /= self.T
            logits_si /= self.T
            logits_it /= self.T
            logits_is /= self.T'''

            logits_ati /= self.T
            logits_asi /= self.T
            logits_ait /= self.T
            logits_ais /= self.T
            '''logits_ati = torch.softmax(logits_ati, dim=1)
            logits_asi = torch.softmax(logits_asi, dim=1)
            logits_ait = torch.softmax(logits_ait, dim=1)
            logits_ais = torch.softmax(logits_ais, dim=1)'''

            '''labels_ddm_ti = logits_ti.clone().detach()
            labels_ddm_ti = torch.softmax(labels_ddm_ti, dim=1)
            labels_ddm_ti = labels_ddm_ti.detach()
            labels_ddm_si = logits_si.clone().detach()
            labels_ddm_si = torch.softmax(labels_ddm_si, dim=1)
            labels_ddm_si = labels_ddm_si.detach()
            labels_ddm_it = logits_it.clone().detach()
            labels_ddm_it = torch.softmax(labels_ddm_it, dim=1)
            labels_ddm_it = labels_ddm_it.detach()
            labels_ddm_is = logits_is.clone().detach()
            labels_ddm_is = torch.softmax(labels_ddm_is, dim=1)
            labels_ddm_is = labels_ddm_is.detach()'''

            '''labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long).cuda()
            labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long).cuda()
            labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long).cuda()
            labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long).cuda()'''
            labels_ati = torch.zeros(logits_ati.shape[0], dtype=torch.long).cuda()
            labels_asi = torch.zeros(logits_asi.shape[0], dtype=torch.long).cuda()
            labels_ait = torch.zeros(logits_ait.shape[0], dtype=torch.long).cuda()
            labels_ais = torch.zeros(logits_ais.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            # self._dequeue_and_enqueue(kt, ks, ki)
            # update memory bank
            total_bsize=1
            momentum = 0.9
            mem_wd = 1e-4
            memory_lr = 3
            mem_t = 0.03
            with torch.no_grad():
                logits = torch.cat([l_pos_ati, l_neg_ati], dim=1) / mem_t
                p_qd=nn.functional.softmax(logits, dim=1)[:,total_bsize:]
                g = torch.einsum('cn,nk->ck',[qat.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg_ati),dim=0),d_norm_ati)
                g = -torch.div(g,torch.norm(d_ati,dim=0))/ mem_t # c*k
                self.i_queue.v.data = momentum * self.i_queue.v.data + g + mem_wd * self.i_queue.W.data
                self.i_queue.W.data = self.i_queue.W.data - memory_lr * self.i_queue.v.data

                logits = torch.cat([l_pos_asi, l_neg_asi], dim=1) / mem_t
                p_qd=nn.functional.softmax(logits, dim=1)[:,total_bsize:]
                g = torch.einsum('cn,nk->ck',[qas.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg_asi),dim=0),d_norm_asi)
                g = -torch.div(g,torch.norm(d_asi,dim=0))/ mem_t # c*k
                self.i_queue.v.data = momentum * self.i_queue.v.data + g + mem_wd * self.i_queue.W.data
                self.i_queue.W.data = self.i_queue.W.data - memory_lr * self.i_queue.v.data

                logits = torch.cat([l_pos_ait, l_neg_ait], dim=1) / mem_t
                p_qd=nn.functional.softmax(logits, dim=1)[:,total_bsize:]
                g = torch.einsum('cn,nk->ck',[qai.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg_ait),dim=0),d_norm_ait)
                g = -torch.div(g,torch.norm(d_ait,dim=0))/ mem_t # c*k
                self.t_queue.v.data = momentum * self.t_queue.v.data + g + mem_wd * self.t_queue.W.data
                self.t_queue.W.data = self.t_queue.W.data - memory_lr * self.t_queue.v.data

                logits = torch.cat([l_pos_ais, l_neg_ais], dim=1) / mem_t
                p_qd=nn.functional.softmax(logits, dim=1)[:,total_bsize:]
                g = torch.einsum('cn,nk->ck',[qai.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg_ais),dim=0),d_norm_ais)
                g = -torch.div(g,torch.norm(d_ais,dim=0))/ mem_t # c*k
                self.s_queue.v.data = momentum * self.s_queue.v.data + g + mem_wd * self.s_queue.W.data
                self.s_queue.W.data = self.s_queue.W.data - memory_lr * self.s_queue.v.data

            return logits_ati, logits_asi, logits_ait, logits_ais, \
                labels_ati, labels_asi, labels_ait, labels_ais, loss_mix
            '''return logits_ti, logits_si, logits_it, logits_is, \
                logits_ati, logits_asi, logits_ait, logits_ais, \
                labels_ti, labels_si, labels_it, labels_is, \
                labels_ddm_ti, labels_ddm_si, labels_ddm_it, labels_ddm_is'''
                
