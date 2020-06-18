# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Written by Feng Zhang & Hong Hu
#
#
# modified by Thomas Verelst
# ESAT-PSI, KU LEUVEN
# ------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.viz as viz
from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds
import dynconv
from utils.flopscounter import (add_flops_counting_methods, flops_to_string,
                                get_model_parameters_number)
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def make_dynconv_meta(config, epoch, iteration):
    if config.DYNCONV.ENABLED:
        return {
            'masks': [],
            'gumbel_temp': 1,
            'gumbel_noise': False,
            'save_masks': True,
            'epoch': epoch,
            'iteration': iteration,
            'gumbel_temp': config.DYNCONV.GUMBEL_TEMP,
            'gumbel_noise': True if epoch < int(config.DYNCONV.NOISE_OFF_EPOCH) else False
        }
    else:
        return {}

def train(config, train_loader, model, criterion, sparsity_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    NUM_ERRORS = 0
    end = time.time()
    train_iter = train_loader.__iter__()
    num_step = len(train_iter)
    for i in range(num_step):
        try:
            # dataloading in try/except for file server is overload
            input, target, target_weight, meta  = next(train_iter)
            NUM_ERRORS = max(0, NUM_ERRORS-1)
        except Exception as e: 
            NUM_ERRORS += 1
            print('Exception at dataloading for train iteration '+str(i)+': '+str(e), end="", flush=True)
            time.sleep(5)
            if NUM_ERRORS > 20:
                raise RuntimeError('Too many dataloader errors')
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        dynconv_meta = make_dynconv_meta(config, epoch, i)
        outputs, dynconv_meta = model(input, dynconv_meta)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        assert isinstance(outputs, list)
        loss = criterion(outputs[0], target, target_weight)
        for output in outputs[1:]:
            loss += criterion(output, target, target_weight)
        output = outputs[-1]


        if config.DYNCONV.ENABLED:
            assert sparsity_criterion is not None
            loss_sparsity, dynconv_meta = sparsity_criterion(dynconv_meta)
            loss = loss + loss_sparsity

            if i % config.PRINT_FREQ == 0:
                sparsity_meta = dynconv_meta['sparsity_meta']
                logger.info(f'train/sparsity_upper_bound: {float(sparsity_meta["upper_bound"])}')
                logger.info(f'train/sparsity_lower_bound: {float(sparsity_meta["lower_bound"])}')
                logger.info(f'train/loss_sparsity_block: {float(sparsity_meta["loss_sp_block"])}')
                logger.info(f'train/loss_sparsity_network: {float(sparsity_meta["loss_sp_network"])}')
                logger.info(f'train/cost: {float(sparsity_meta["cost_perc"])}')
                logger.info(f'train/loss_sparsity: {float(loss_sparsity)}')
                logger.info(f'train/loss: {float(loss)}')

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, epoch, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    logger.info(f'# VALIDATE: EPOCH {epoch}')

    model = add_flops_counting_methods(model)
    model.start_flops_count()
    model.eval()

    flops_per_layer = []
    total_per_layer = []

    with torch.no_grad():
        end = time.time()
        val_iter = val_loader.__iter__()
        num_step = len(val_iter)
        for i in range(num_step):
            input, target, target_weight, meta  = next(val_iter)
            input = input.to('cuda', non_blocking=True)

            dynconv_meta = make_dynconv_meta(config, epoch, i)
            outputs, dynconv_meta = model(input, dynconv_meta)
            
            if 'masks' in dynconv_meta:
                percs, cost, total = dynconv.cost_per_layer(dynconv_meta)
                flops_per_layer.append(cost)
                total_per_layer.append(total)
    

            output = outputs[-1] if isinstance(outputs, list) else outputs
            
            # if config.TEST.FLIP_TEST:
            # flip not supported for dynconv
            #     # this part is ugly, because pytorch has not supported negative index
            #     # input_flipped = model(input[:, :, :, ::-1])
            #     input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            #     input_flipped = torch.from_numpy(input_flipped).cuda()
            #     outputs_flipped = model(input_flipped)

            #     if isinstance(outputs_flipped, list):
            #         output_flipped = outputs_flipped[-1]
            #     else:
            #         output_flipped = outputs_flipped

            #     output_flipped = flip_back(output_flipped.cpu().numpy(),
            #                                val_dataset.flip_pairs)
            #     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            #     # feature is not aligned, shift flipped heatmap for higher accuracy
            #     if config.TEST.SHIFT_HEATMAP:
            #         output_flipped[:, :, :, 1:] = \
            #             output_flipped.clone()[:, :, :, 0:-1]

            #     output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            output_np = output.clone().cpu().numpy()
            preds_rel, maxvals_rel = get_max_preds(output_np)
            preds, maxvals = get_final_preds(
                config, output_np, c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )

                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

            if config.DEBUG.PONDER:
                img = viz.frame2mpl(input[0], denormalize=True)
                img = viz.add_skeleton(img, preds_rel[0]*4, maxvals_rel[0], thres=0.2)

                plt.figure()
                plt.title('input')
                plt.imshow(img)
                ponder_cost = dynconv.ponder_cost_map(dynconv_meta['masks'])
                if ponder_cost is not None:
                    plt.figure()
                    plt.title('ponder cost map')
                    plt.imshow(ponder_cost, vmin=2, vmax=len(dynconv_meta['masks'])-2)
                    plt.colorbar()
                else:
                    logger.info('Not a sparse model - no ponder cost')
                viz.showKey()

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    avg_flops, total_flops, batch_count = model.compute_average_flops_cost()
    logger.info(f'# PARAMS: {get_model_parameters_number(model, as_string=False)/1e6} M')
    logger.info(f'# average FLOPS (multiply-accumulates, MACs) per image: {(total_flops/idx)/1e9} GMacs on {idx} images')

    # some conditional execution statistics
    if len(flops_per_layer) > 0:
        flops_per_layer = torch.cat(flops_per_layer, dim=0)
        total_per_layer = torch.cat(total_per_layer, dim=0)

        perc_per_layer = flops_per_layer/total_per_layer

        perc_per_layer_avg = perc_per_layer.mean(dim=0)
        perc_per_layer_std = perc_per_layer.std(dim=0)
        
        s = ''
        for perc in perc_per_layer_avg:
            s += f'{round(float(perc), 2)}, '
        logger.info(f'# average FLOPS (multiply-accumulates MACs) used percentage per layer (average): {s}')
        
        s = ''
        for std in perc_per_layer_std:
            s += f'{round(float(std), 2)}, '
        logger.info(f'# average FLOPS (multiply-accumulates MACs) used percentage per layer (standard deviation): {s}')


        exec_cond_flops = int(torch.sum(flops_per_layer))/idx
        total_cond_flops = int(torch.sum(total_per_layer))/idx
        logger.info(f'# Conditional average FLOPS (multiply-accumulates MACs) over all layers (average per image): {exec_cond_flops/1e9} GMac out of {total_cond_flops/1e9} GMac ({round(100*exec_cond_flops/total_cond_flops,1)}%)')

    return perf_indicator


def speedtest(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, epoch, writer_dict=None):
    '''
    Speedtest mode first warms up on half the test size (especially Pytorch
    CUDA benchmark mode needs warmup to optimize operations), 
    and then performs the speedtest on the other half
    '''

    # switch to evaluate mode
    model.eval()

    idx = 0

    logger.info(f'# SPEEDTEST: EPOCH {epoch}')

    logger.info('\n\n>> WARMUP')
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    with torch.no_grad():
        val_iter = val_loader.__iter__()
        num_step = len(val_iter)
        for i in range(num_step):

            if i == num_step//2:
                avg_flops, total_flops, batch_count = model.compute_average_flops_cost()
                logger.info(f'# PARAMS {get_model_parameters_number(model, as_string=False)/1e6}M')
                logger.info(f'# FLOPS (multiply-accumulates, MACs): {(total_flops/idx)/1e9} G on {idx} images (batch_count={batch_count})')
                model.stop_flops_count()
                idx = 0
                logger.info('\n\n>> SPEEDTEST')
                torch.cuda.synchronize()
                START = time.perf_counter()

            input, _, _, _  = next(val_iter)
            input = input.cuda(non_blocking=True)
            dynconv_meta = make_dynconv_meta(config, epoch, i)
            outputs, dynconv_meta = model(input, dynconv_meta)

            output = outputs[-1] if isinstance(outputs, list) else outputs
            if config.TEST.FLIP_TEST:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                        val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)
            idx += num_images

    torch.cuda.synchronize()
    STOP = time.perf_counter()
    samples_per_second = idx/(STOP-START)
    logger.info(f'ELAPSED TIME: {(STOP-START)}s, SAMPLES PER SECOND: {samples_per_second} ON {idx} SAMPLES')

    return idx/(STOP-START)


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
