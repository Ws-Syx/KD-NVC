# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Adapted for three-version (standard / small / tiny) inference on WSL GPU.
# No CUDA_INFERENCE custom kernels - standard PyTorch forward only.

import argparse
import io
import json
import os
import time

import torch
import numpy as np

# Suppress warning about missing custom CUDA kernels
os.environ.setdefault('SUPPRESS_CUSTOM_KERNEL_WARNING', '1')

from src.layers.cuda_inference import replicate_pad
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, \
    dump_json, set_torch_env
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.video_reader import YUV420Reader
from src.utils.metrics import calc_psnr
from src.utils.transforms import yuv_444_to_420, ycbcr420_to_444_np


# ---------------------------------------------------------------------------
# Model registry: --model_type maps to import path + checkpoint file
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    'standard': {
        'module': 'src.models.video_model',
        'checkpoint': 'checkpoint/cvpr2025_video.pth.tar',
    },
    'small': {
        'module': 'src.models.video_model_small',
        'checkpoint': 'checkpoint/kd-nvc-small-stage8-epoch29.tar',
    },
    'tiny': {
        'module': 'src.models.video_model_tiny',
        'checkpoint': 'checkpoint/kd-nvc-tiny-stage8-epoch26.pth.tar',
    },
}

I_FRAME_CHECKPOINT = 'checkpoint/cvpr2025_image.pth.tar'


def parse_args():
    parser = argparse.ArgumentParser(description="DCVC-RT three-version inference test")

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['standard', 'small', 'tiny'],
                        help='P-frame model variant')
    parser.add_argument('--force_zero_thres', type=float, default=0.12)
    parser.add_argument('--rate_num', type=int, default=1)
    parser.add_argument('--qp', type=int, default=32)
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=4)
    parser.add_argument("--force_intra_period", type=int, default=None,
                        help='I-frame period: N>0 -> an I-frame every N frames, '
                             '-1/0 -> a single I-frame at frame 0. '
                             'Default: keep intra_period from --test_config')
    parser.add_argument('--reset_interval', type=int, default=64,
                        help='feature-adaptor-I refresh period (DCVC-RT reset_interval); '
                             '-1/0 disables the refresh')
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument('--calc_ssim', type=str2bool, default=False)
    parser.add_argument('--write_stream', type=str2bool, default=True)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--random_weights', type=str2bool, default=False,
                        help='do not load checkpoints; use random init for speed-only benchmark')

    args = parser.parse_args()
    args.qp_i = [args.qp]
    args.qp_p = [args.qp]
    return args


# ---------------------------------------------------------------------------
# Frame I/O helpers
# ---------------------------------------------------------------------------
def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device, dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    assert args['src_type'] == 'yuv420'
    return YUV420Reader(args['src_path'], args['src_width'], args['src_height'])


def get_src_frame(args, src_reader, device):
    y, uv = src_reader.read_one_frame()
    yuv = ycbcr420_to_444_np(y, uv)
    x = np_image_to_tensor(yuv, device)
    y_plane = y[0, :, :]
    u_plane = uv[0, :, :]
    v_plane = uv[1, :, :]
    x = x.to(torch.float16)
    return x, y_plane, u_plane, v_plane


def get_distortion(args, x_hat, y, u, v):
    y_rec, uv_rec = yuv_444_to_420(x_hat)
    y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    y_r = y_rec[0, :, :]
    u_r = uv_rec[0, :, :]
    v_r = uv_rec[1, :, :]
    psnr_y = calc_psnr(y, y_r)
    psnr_u = calc_psnr(u, u_r)
    psnr_v = calc_psnr(v, v_r)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
    return [psnr, psnr_y, psnr_u, psnr_v]


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def init_models(args, device):
    """Create both I-frame and P-frame models.

    With --random_weights, checkpoints are NOT loaded: the models keep their
    random initialization, which is sufficient for pure speed benchmarking
    (architecture + CUDA kernels are identical; weight values do not affect
    timing). Rate/PSNR outputs are then meaningless.
    """
    set_torch_env()

    model_cfg = MODEL_REGISTRY[args.model_type]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- I-frame model (shared) ---
    i_frame_net = DMCI()
    if args.random_weights:
        print("[WARN] random_weights=True: I-frame checkpoint NOT loaded (speed-only)")
    else:
        i_path = os.path.join(base_dir, I_FRAME_CHECKPOINT)
        i_state_dict = get_state_dict(i_path)
        i_frame_net.load_state_dict(i_state_dict)
        print(f"[OK] I-frame model loaded: {I_FRAME_CHECKPOINT}")
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    i_frame_net.update(args.force_zero_thres)
    i_frame_net.half()

    # --- P-frame model ---
    import importlib
    mod = importlib.import_module(model_cfg['module'])
    DMC = mod.DMC
    p_frame_net = DMC()
    if args.random_weights:
        print(f"[WARN] random_weights=True: P-frame checkpoint NOT loaded (speed-only) [{args.model_type}]")
    else:
        p_path = os.path.join(base_dir, model_cfg['checkpoint'])
        p_state_dict = get_state_dict(p_path)
        p_frame_net.load_state_dict(p_state_dict)
        print(f"[OK] P-frame model loaded [{args.model_type}]: {model_cfg['checkpoint']}")
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()
    p_frame_net.update(args.force_zero_thres)
    p_frame_net.half()

    return i_frame_net, p_frame_net


# ---------------------------------------------------------------------------
# Compress -> Stream -> Decompress pipeline (adapted from official test_video.py)
# ---------------------------------------------------------------------------
def run_one_point(p_frame_net, i_frame_net, args):
    if args['check_existing'] and os.path.exists(args['curr_json_path']):
        with open(args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result['i_frame_num'] + log_result['p_frame_num'] == args['frame_num']:
                return log_result
            print(f"Incomplete log for {args['curr_json_path']}, rerunning.")

    frame_num = args['frame_num']
    verbose = args['verbose']
    intra_period = args['intra_period']
    device = next(i_frame_net.parameters()).device

    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    use_two_entropy_coders = pic_height * pic_width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    frame_types = []
    psnrs = []
    bits = []
    encoding_times = []
    decoding_times = []

    start_time = time.time()
    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    p_frame_net.set_curr_poc(0)
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    # ================================================================ ENCODE
    with torch.no_grad():
        last_qp = 0
        for frame_idx in range(frame_num):
            x, y, u, v = get_src_frame(args, src_reader, device)

            torch.cuda.synchronize(device=device)
            t0 = time.time()

            x_padded = replicate_pad(x, padding_b, padding_r)

            is_i_frame = (frame_idx == 0 or
                          (intra_period > 0 and frame_idx % intra_period == 0))

            if is_i_frame:
                curr_qp = args['qp_i']
                sps = {'sps_id': -1, 'height': pic_height, 'width': pic_width,
                       'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0}
                encoded = i_frame_net.compress(x_padded, curr_qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
                frame_types.append(0)
            else:
                fa_idx = index_map[frame_idx % 8]
                if args['reset_interval'] > 0 and frame_idx % args['reset_interval'] == 1:
                    use_ada_i = 1
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                else:
                    use_ada_i = 0
                curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)
                sps = {'sps_id': -1, 'height': pic_height, 'width': pic_width,
                       'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': use_ada_i}
                encoded = p_frame_net.compress(x_padded, curr_qp)
                last_qp = curr_qp
                frame_types.append(1)

            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            sps_bytes = write_sps(output_buff, sps) if sps_new else 0
            stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp,
                                    encoded['bit_stream'])
            bits.append(stream_bytes * 8 + sps_bytes * 8)

            torch.cuda.synchronize(device=device)
            t1 = time.time()
            encoding_times.append(t1 - t0)

            if verbose >= 2:
                print(f"  enc frame {frame_idx:3d}  {'I' if is_i_frame else 'P'}  "
                      f"{(t1 - t0) * 1000:7.2f} ms  bits={bits[-1]:,}")

    src_reader.close()

    total_bytes = output_buff.getbuffer().nbytes
    total_kbps = int(total_bytes * 8 / (frame_num / 30) / 1000)

    # Write stream and re-read for decode
    with open(args['curr_bin_path'], "wb") as f:
        f.write(output_buff.getbuffer())
    output_buff.close()

    # ================================================================ DECODE
    sps_helper = SPSHelper()
    with open(args['curr_bin_path'], "rb") as f:
        input_buff = io.BytesIO(f.read())
    src_reader = get_src_reader(args)

    p_frame_net.set_curr_poc(0)
    decoded_count = 0
    with torch.no_grad():
        while decoded_count < frame_num:
            x, y, u, v = get_src_frame(args, src_reader, device)

            torch.cuda.synchronize(device=device)
            t0 = time.time()

            header = read_header(input_buff)
            while header['nal_type'] == NalType.NAL_SPS:
                sps_obj = read_sps_remaining(input_buff, header['sps_id'])
                sps_helper.add_sps_by_id(sps_obj)
                if verbose >= 2:
                    print(f"  new sps: {sps_obj}")
                header = read_header(input_buff)

            sps_obj = sps_helper.get_sps_by_id(header['sps_id'])
            decode_qp, bit_stream = read_ip_remaining(input_buff)

            dec_is_i = (header['nal_type'] == NalType.NAL_I)
            if dec_is_i:
                decoded = i_frame_net.decompress(bit_stream, sps_obj, decode_qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, decoded['x_hat'])
            else:
                if sps_obj.get('use_ada_i', 0):
                    p_frame_net.reset_ref_feature()
                decoded = p_frame_net.decompress(bit_stream, sps_obj, decode_qp)

            x_hat = decoded['x_hat'][:, :, :pic_height, :pic_width]

            torch.cuda.synchronize(device=device)
            t1 = time.time()
            decoding_times.append(t1 - t0)

            curr_psnr = get_distortion(args, x_hat, y, u, v)
            psnrs.append(curr_psnr)

            if verbose >= 2:
                stream_len = 0 if bit_stream is None else len(bit_stream) * 8
                print(f"  dec frame {decoded_count:3d}  {'I' if dec_is_i else 'P'}  "
                      f"{(t1 - t0) * 1000:7.2f} ms  bits={stream_len:,}  "
                      f"PSNR={curr_psnr[0]:.3f}")

            decoded_count += 1

    input_buff.close()
    src_reader.close()

    # ================================================================ REPORT
    total_time = time.time() - start_time
    avg_enc = sum(encoding_times) / len(encoding_times)
    avg_dec = sum(decoding_times) / len(decoding_times)
    avg_psnr = np.mean([p[0] for p in psnrs])

    print(f"\n  {'='*55}")
    print(f"  Model: {args['model_type']}  |  {pic_width}x{pic_height}  |  "
          f"{frame_num} frames  |  QP={args['qp_i']}")
    print(f"  {'='*55}")
    print(f"  intra_period={intra_period}  reset_interval={args['reset_interval']}")
    print(f"  Total bytes : {total_bytes:,}  ({total_kbps} kbps @ 30fps)")
    print(f"  Avg PSNR    : {avg_psnr:.4f} dB")
    print(f"  Enc avg     : {avg_enc*1000:.1f} ms/f  ({1/avg_enc:.1f} FPS)")
    print(f"  Dec avg     : {avg_dec*1000:.1f} ms/f  ({1/avg_dec:.1f} FPS)")
    print(f"  Total time  : {total_time:.1f} s")

    log_result = generate_log_json(
        frame_num, pic_height * pic_width, total_time,
        frame_types, bits, psnrs, psnrs,  # psnrs repeated for msssim placeholder
        verbose=False,
        avg_encoding_time=avg_enc,
        avg_decoding_time=avg_dec,
    )

    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)

    return log_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    begin_time = time.time()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    # Device selection
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU]")

    # Load config
    with open(args.test_config) as f:
        config = json.load(f)

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']

    # Init models
    i_frame_net, p_frame_net = init_models(args, device)

    # Process each sequence
    results = []
    test_classes = config['test_classes']
    for ds_name in test_classes:
        if test_classes[ds_name].get('test', 1) == 0:
            continue
        for seq_name in test_classes[ds_name]['sequences']:
            seq_cfg = test_classes[ds_name]['sequences'][seq_name]

            cur_args = {
                'rate_idx': 0,
                'qp_i': args.qp_i[0],
                'qp_p': args.qp_p[0],
                'force_intra': args.force_intra,
                'reset_interval': args.reset_interval,
                'seq': seq_name,
                'model_type': args.model_type,
                'src_type': test_classes[ds_name]['src_type'],
                'src_height': seq_cfg['height'],
                'src_width': seq_cfg['width'],
                'intra_period': seq_cfg.get('intra_period', -1),
                'frame_num': seq_cfg['frames'],
                'calc_ssim': args.calc_ssim,
                'dataset_path': os.path.join(root_path, test_classes[ds_name]['base_path']),
                'write_stream': args.write_stream,
                'check_existing': args.check_existing,
                'stream_path': args.stream_path,
                'save_decoded_frame': args.save_decoded_frame,
                'ds_name': ds_name,
                'verbose': args.verbose,
            }

            if args.force_frame_num > 0:
                cur_args['frame_num'] = args.force_frame_num
            if args.force_intra:
                cur_args['intra_period'] = 1
            if args.force_intra_period is not None:
                cur_args['intra_period'] = args.force_intra_period

            print(f"[SEQ] {ds_name}/{seq_name}  "
                  f"{cur_args['src_width']}x{cur_args['src_height']}  "
                  f"frames={cur_args['frame_num']}  "
                  f"intra_period={cur_args['intra_period']}  "
                  f"reset_interval={cur_args['reset_interval']}  "
                  f"qp={args.qp}")

            # Setup output paths
            bin_folder = os.path.join(args.stream_path, ds_name)
            create_folder(bin_folder, True)
            cur_args['src_path'] = os.path.join(cur_args['dataset_path'], seq_name)
            cur_args['bin_folder'] = bin_folder
            cur_args['curr_bin_path'] = os.path.join(bin_folder, f"{seq_name}_q{args.qp}.bin")
            cur_args['curr_rec_path'] = cur_args['curr_bin_path'].replace('.bin', '.yuv')
            cur_args['curr_json_path'] = cur_args['curr_bin_path'].replace('.bin', '.json')

            result = run_one_point(p_frame_net, i_frame_net, cur_args)
            result['ds_name'] = ds_name
            result['seq'] = seq_name
            result['rate_idx'] = 0
            result['qp_i'] = args.qp_i[0]
            result['qp_p'] = args.qp_p[0]
            results.append(result)

    # Write summary JSON
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        create_folder(out_dir, True)
    log_result = {'model_type': args.model_type, 'results': {}}
    for res in results:
        log_result['results'][res['ds_name']] = {res['seq']: {f"{res['rate_idx']:03d}": res}}
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    elapsed = (time.time() - begin_time) / 60
    print(f"\nTest finished [{args.model_type}].  Elapsed: {elapsed:.1f} min")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
