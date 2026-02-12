import torch 
import random
from utils import psnr_error
from eval import val
from training.train_ing_func import *
import time
import datetime
from training.losses import ramp, temporal_consistency_loss, bezier_trajectory_loss



def training(cfg, dataset, dataloader, models, losses, opts, scores):
    # define start_iter
    start_iter = scores['step'] if scores['step'] > 0 else 0

    # find epoch 
    epoch = int(scores['step']/len(dataloader)) # [epoch: current step / (data size / batch size)] ex) 8/(16/4)

    # training start!
    training = True  
    torch.autograd.set_detect_anomaly(True)

    print('\n===========================================================')
    print('Training Start!')
    print('===========================================================')

    while training:
        '''
        ------------------------------------
        Training (1 epoch)
        ------------------------------------
        '''
        for indice, clips in dataloader:
            # define frame 1 to 4 
            frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 3, 256, 256) 
            frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 3, 256, 256) 
            frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 3, 256, 256) 
            frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 3, 256, 256) 

            # pop() the used video index
            for index in indice:
                dataset.all_seqs[index].pop()
                if len(dataset.all_seqs[index]) == 0:
                    dataset.all_seqs[index] = list(range(len(dataset.videos[index]) - 4))
                    random.shuffle(dataset.all_seqs[index])

            # generator input, target
            input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (n, 12, 256, 256) 
            target = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256) 

            # forward
            G_l, D_l, F_frame, temp_raw, bez_raw, lam_t, lam_b = forward(
                input=input.cuda(),
                target=target,
                input_last=frame_4,
                frame_1=frame_1, frame_2=frame_2, frame_3=frame_3, frame_4=frame_4,
                step=scores['step'],
                cfg=cfg,
                models=models,
                losses=losses
            )

            scores['g_loss_list'].append(G_l.item())
            scores['d_loss_list'].append(D_l.item())
            scores.setdefault('temp_list', []).append(float(temp_raw.item()))
            scores.setdefault('bez_list', []).append(float(bez_raw.item()))
            scores.setdefault('lam_t_list', []).append(float(lam_t))
            scores.setdefault('lam_b_list', []).append(float(lam_b))


            # backward
            opts['optimizer_G'].zero_grad()
            G_l.backward()

            if (lam_t > 0 or lam_b > 0) and (scores['step'] % 1000 == 0):
                print("[DEBUG] lam_t, lam_b:", lam_t, lam_b)
                print("[DEBUG] temp_raw grad:", temp_raw.requires_grad, temp_raw.grad_fn)
                print("[DEBUG] bez_raw  grad:", bez_raw.requires_grad, bez_raw.grad_fn)
                g = next(models['generator'].parameters()).grad
                print("[DEBUG] gen grad None?", g is None, "mean abs:", (g.abs().mean().item() if g is not None else None))

            opts['optimizer_G'].step()

            opts['optimizer_D'].zero_grad()
            D_l.backward()
            opts['optimizer_D'].step()

            # calculate time
            torch.cuda.synchronize()
            time_end = time.time()
            if scores['step'] > start_iter:  
                iter_t = time_end - temp
            temp = time_end


            if scores['step'] != start_iter:
                '''
                -----------------------------------
                check train status per 20 iteration
                -----------------------------------
                '''
                if scores['step'] % 20 == 0:
                    print(f"===========epoch:{epoch} (step:{scores['step']})============")

                    # calculate remained time
                    time_remain = (cfg.iters - scores['step']) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]

                    # calculate psnr
                    psnr = psnr_error(F_frame, target)

                    # print loss, psnr, auc
                    print(f"[{scores['step']}] G_l: {G_l:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                        f"temp_l: {float(temp_raw):.4f} | bez_l: {float(bez_raw):.4f} | "
                        f"lam_t: {lam_t:.12f} | lam_b: {lam_b:.12f} | "
                        f"best_auc: {scores['best_auc']:.3f} | iter_t: {iter_t:.3f}s | remain_t: {eta}"
                    )

                    print("[DEBUG] lam_t, lam_b:", float(lam_t), float(lam_b))

                    # view loss by graph
                    view_loss(cfg, scores)

                '''
                --------------------------------
                find Best model per val_interval
                --------------------------------
                '''
                if scores['step'] % cfg.val_interval == 0:
                    auc, scores = val(cfg=cfg, train_scores=scores, models=models, iter=scores['step'])
                    update_best_model(cfg, auc, scores['step'], models, opts, scores)

                '''
                ------------------------------------
                save current model per save_interval 
                ------------------------------------
                '''
                if scores['step'] % cfg.save_interval == 0:
                    model_dict = make_models_dict(models, opts, scores)
                    torch.save(model_dict, f'weights/latest_{cfg.dataset}.pth')
                    print(f"\nAlready saved: \'latest_{cfg.dataset}.pth\'.")

                # training complete!
                if scores['step'] == cfg.iters:
                    training = False

            # one iteration ok!
            scores['step'] += 1
            
        # one epoch ok!
        epoch += 1
        

def forward(input, target, input_last, frame_1, frame_2, frame_3, frame_4, step, cfg, models, losses):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    generator = models['generator']
    discriminator = models['discriminator']
    flownet = models['flownet']

    discriminate_loss = losses['discriminate_loss']
    intensity_loss = losses['intensity_loss']
    gradient_loss = losses['gradient_loss']
    adversarial_loss = losses['adversarial_loss']
    flow_loss = losses['flow_loss']

    coefs = [1, 1, 0.05, 2] # inte_l, grad_l, adv_l, flow_l

    # future frame prediction and get loss
    pred  = generator(input)
    inte_l = intensity_loss(pred, target)
    grad_l = gradient_loss(pred, target)
    adv_l = adversarial_loss(discriminator(pred))

    # flowmap prediction and get loss
    gt_flow_input = torch.cat([input_last.unsqueeze(2), target.unsqueeze(2)], 2)
    pred_flow_input = torch.cat([input_last.unsqueeze(2), pred.unsqueeze(2)], 2)

    flow_gt = (flownet(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
    flow_pred = (flownet(pred_flow_input * 255.) / 255.).detach()
    flow_l = flow_loss(flow_pred, flow_gt)


    #--# =========================
    # B3: Temporal + Bezier
    # =========================
    lam_t = ramp(step, cfg.temp_ramp_start, cfg.temp_ramp_end, cfg.lambda_temporal_max) if cfg.use_temporal else 0.0
    lam_b = ramp(step, cfg.bez_ramp_start,  cfg.bez_ramp_end,  cfg.lambda_bezier_max)   if cfg.use_bezier else 0.0

    do_temp = (cfg.use_temporal and (step >= cfg.temp_ramp_start) and (lam_t > 0.0))
    do_bez  = (cfg.use_bezier   and (step >= cfg.bez_ramp_start)  and (lam_b > 0.0))

    temp_raw = torch.tensor(0.0, device=pred.device)
    bez_raw  = torch.tensor(0.0, device=pred.device)
    temp_l   = torch.tensor(0.0, device=pred.device)
    bez_l    = torch.tensor(0.0, device=pred.device)

    # Temporal only when active
    if do_temp:
        temp_raw = temporal_consistency_loss(pred, frame_4, flow_gt)
        temp_l = lam_t * temp_raw

    # Bezier only when active (extra FlowNet calls happen ONLY here)
    if do_bez:
        # 12-23-34: GT akışlar (detach kalabilir)
        f12_in = torch.cat([frame_1.unsqueeze(2), frame_2.unsqueeze(2)], 2)
        f23_in = torch.cat([frame_2.unsqueeze(2), frame_3.unsqueeze(2)], 2)
        f34_in = torch.cat([frame_3.unsqueeze(2), frame_4.unsqueeze(2)], 2)

        flow_12 = (flownet(f12_in * 255.) / 255.).detach()
        flow_23 = (flownet(f23_in * 255.) / 255.).detach()
        flow_34 = (flownet(f34_in * 255.) / 255.).detach()

        # 45: pred'e bağlı olmalı (DETACH YOK!)  IMPORTANT: last flow MUST depend on pred (NO detach) => grad flows to generator
        f45_pred_in = torch.cat([frame_4.unsqueeze(2), pred.unsqueeze(2)], 2)
        flow_45_pred = (flownet(f45_pred_in * 255.) / 255.)   # <-- detach YOK

        bez_raw = bezier_trajectory_loss(flow_12, flow_23, flow_34, flow_45_pred)
        bez_l = lam_b * bez_raw

        
    # DEBUG: only print when losses are active (avoid spam)
    if (do_temp or do_bez) and (step % 1000 == 0):
        print("[DEBUG2] temp_l reqgrad:", temp_l.requires_grad, "grad_fn:", temp_l.grad_fn)
        print("[DEBUG2] bez_l  reqgrad:", bez_l.requires_grad,  "grad_fn:", bez_l.grad_fn)

    if do_bez and (step % 1000 == 0):
        print("[CHECK] bez_raw reqgrad:", bez_raw.requires_grad, "grad_fn:", bez_raw.grad_fn)
    

    loss_gen = coefs[0] * inte_l + \
            coefs[1] * grad_l + \
            coefs[2] * adv_l + \
            coefs[3] * flow_l + \
            temp_l + \
            bez_l

    # discriminator
    loss_dis = discriminate_loss(discriminator(target),
                                 discriminator(pred.detach()))

    return loss_gen, loss_dis, pred, temp_raw.detach(), bez_raw.detach(), lam_t, lam_b

