import json
import os
import subprocess
import sys
from functools import partial
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


# RANK = int(os.environ.get('RANK', 0))
# WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
# LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
# LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

# GPU_LIST = get_gpu_list()
# if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
#     NGPU = len(GPU_LIST)
#     assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
#     GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
#     DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
#     CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
#     CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
#     # Set CUDA_VISIBLE_DEVICES
#     os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
#     print(
#         f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
#         f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
#     )
def setup_distributed_env():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ['EVAL_VLLM_OUTER_RANK'] = str(rank)
    os.environ['EVAL_VLLM_OUTER_WORLD_SIZE'] = str(world_size)
    os.environ['EVAL_VLLM_OUTER_LOCAL_RANK'] = str(local_rank)
    os.environ['EVAL_VLLM_OUTER_LOCAL_WORLD_SIZE'] = str(local_world_size)

    gpu_list = get_gpu_list()
    if local_world_size > 1 and len(gpu_list):
        ngpu = len(gpu_list)
        assert ngpu >= local_world_size, "The number of processes should be less than or equal to the number of GPUs"
        gpu_per_proc = ngpu // local_world_size
        device_start_idx = gpu_per_proc * local_rank
        cuda_visible_devices = ','.join(str(i) for i in gpu_list[device_start_idx: device_start_idx + gpu_per_proc])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        print(
            f'RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}, '
            f'LOCAL_WORLD_SIZE: {local_world_size}, CUDA_VISIBLE_DEVICES: {cuda_visible_devices}'
        )

    host_ip = os.environ.get('VLLM_HOST_IP') or os.environ.get('MASTER_ADDR') or os.environ.get('HOST_IP')
    if host_ip and host_ip not in ('127.0.0.1', 'localhost'):
        os.environ.setdefault('VLLM_HOST_IP', host_ip)
        # vLLM's single-node multiproc executor otherwise uses 127.0.0.1.
        os.environ.setdefault('VLLM_LOOPBACK_IP', host_ip)

    base_vllm_port = int(os.environ.get('VLLM_PORT', os.environ.get('VLLM_PORT_BASE', 20000)))
    if local_world_size > 1:
        os.environ['VLLM_PORT'] = str(base_vllm_port + local_rank * 100)
    else:
        os.environ.setdefault('VLLM_PORT', str(base_vllm_port))

    print(
        f'RANK: {rank}, LOCAL_RANK: {local_rank}, VLLM_HOST_IP={os.environ.get("VLLM_HOST_IP", "")}, '
        f'VLLM_LOOPBACK_IP={os.environ.get("VLLM_LOOPBACK_IP", "")}, '
        f'VLLM_PORT={os.environ.get("VLLM_PORT", "")}'
    )

    return rank, world_size, local_world_size, local_rank


# Bind each worker to its remapped visible device before importing torch/vlmeval.
RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, LOCAL_RANK = setup_distributed_env()

from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.debug_logger import eval_vllm_log, eval_vllm_log_exception
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


# Make WORLD_SIZE invisible when build models
# def build_model_from_config(cfg, model_name, use_vllm=False):
#     import vlmeval.api
#     import vlmeval.vlm
#     ws_bak = os.environ.pop('WORLD_SIZE', None)

#     config = cp.deepcopy(cfg[model_name])
#     if use_vllm:
#         config['use_vllm'] = use_vllm
#     if 'class' not in config:
#         return supported_VLM[model_name](**config)
#     cls_name = config.pop('class')
#     if hasattr(vlmeval.api, cls_name):
#         model = getattr(vlmeval.api, cls_name)(**config)
#     elif hasattr(vlmeval.vlm, cls_name):
#         model = getattr(vlmeval.vlm, cls_name)(**config)
#     else:
#         raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')

#     if ws_bak:
#         os.environ['WORLD_SIZE'] = ws_bak
#     return model
def build_model_from_config(cfg, model_name, use_vllm=False):
    import vlmeval.api
    import vlmeval.vlm
    # Hide torchrun's distributed env so vLLM manages its own TP process group.
    torchrun_env_keys = {
        k for k in os.environ
        if k.startswith('TORCHELASTIC_')
    }
    torchrun_env_keys.update(
        (
            'WORLD_SIZE',
            'RANK',
            'LOCAL_RANK',
            'LOCAL_WORLD_SIZE',
            'MASTER_ADDR',
            'MASTER_PORT',
            'GROUP_RANK',
            'ROLE_RANK',
            'ROLE_WORLD_SIZE',
        )
    )
    env_bak = {
        k: os.environ.pop(k, None)
        for k in torchrun_env_keys
    }

    try:
        config = cp.deepcopy(cfg[model_name])
        eval_vllm_log(
            'run.build_model_from_config_start',
            model_name=model_name,
            use_vllm=use_vllm,
            config=config,
            hidden_env_keys=sorted(k for k, v in env_bak.items() if v is not None),
        )
        if use_vllm:
            config['use_vllm'] = use_vllm
        if 'class' not in config:
            model = supported_VLM[model_name](**config)
            eval_vllm_log('run.build_model_from_config_done', model_name=model_name)
            return model

        cls_name = config.pop('class')
        if hasattr(vlmeval.api, cls_name):
            model = getattr(vlmeval.api, cls_name)(**config)
        elif hasattr(vlmeval.vlm, cls_name):
            model = getattr(vlmeval.vlm, cls_name)(**config)
        else:
            raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')
        eval_vllm_log('run.build_model_from_config_done', model_name=model_name, cls_name=cls_name)
        return model
    except Exception as e:
        eval_vllm_log_exception('run.build_model_from_config_failed', e, model_name=model_name)
        raise
    finally:
        for k, v in env_bak.items():
            if v is not None:
                os.environ[k] = v



def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    cls_name = {
        'Video-MME': 'VideoMME',
    }.get(cls_name, cls_name)
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        if 'dataset' in sig.parameters and 'dataset' not in config:
            config['dataset'] = dataset_name
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=int, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')

    args = parser.parse_args()
    return args


def main():
    global RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, LOCAL_RANK
    logger = get_logger('RUN')
    args = parse_args()
    eval_vllm_log(
        'run.main_start',
        argv=sys.argv,
        rank=RANK,
        world_size=WORLD_SIZE,
        local_rank=LOCAL_RANK,
        local_world_size=LOCAL_WORLD_SIZE,
        use_vllm=args.use_vllm,
        work_dir=args.work_dir,
    )
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
        eval_vllm_log(
            'run.config_loaded',
            config=args.config,
            models=args.model,
            datasets=args.data,
        )
    else:
        assert len(args.data), '--data should be a list of data files'

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from vlmeval.config import api_models as supported_APIs
            from vlmeval.api import GPT4V
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')

    dist = None
    dist_initialized = False
    if WORLD_SIZE > 1:
        import torch.distributed as dist

    def ensure_dist_initialized():
        nonlocal dist_initialized
        if WORLD_SIZE <= 1 or dist_initialized:
            return
        backend = os.environ.get('VLMEVAL_DIST_BACKEND')
        if backend is None:
            backend = 'gloo' if args.use_vllm else 'nccl'
        if RANK == 0:
            logger.info(f'Initializing outer torchrun process group with backend={backend}')
        eval_vllm_log('run.dist_init_start', backend=backend)
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )
        dist_initialized = True
        eval_vllm_log('run.dist_init_done', backend=backend)

    def traced_barrier(label):
        if WORLD_SIZE <= 1:
            return
        eval_vllm_log('run.barrier_start', label=label)
        dist.barrier()
        eval_vllm_log('run.barrier_done', label=label)

    if WORLD_SIZE > 1 and not args.use_vllm:
        ensure_dist_initialized()

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            logger.info(
                f'Rank {RANK}/{WORLD_SIZE}: building model {model_name} '
                f'(use_vllm={args.use_vllm}, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "")})'
            )
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm)
            logger.info(f'Rank {RANK}/{WORLD_SIZE}: finished building model {model_name}')

        if WORLD_SIZE > 1:
            ensure_dist_initialized()

        for _, dataset_name in enumerate(args.data):
            traced_barrier(f'before_dataset:{model_name}:{dataset_name}')

            try:
                pred_format = get_pred_file_format()
                result_file_base = f'{model_name}_{dataset_name}.{pred_format}'
                eval_vllm_log(
                    'run.dataset_start',
                    model_name=model_name,
                    dataset_name=dataset_name,
                    pred_root=pred_root,
                    result_file_base=result_file_base,
                )

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            eval_vllm_log('run.dataset_build_rank0_prepare_start', dataset_name=dataset_name)
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                            eval_vllm_log('run.dataset_build_rank0_prepare_done', dataset_name=dataset_name)
                        traced_barrier(f'after_rank0_dataset_prepare:{dataset_name}')
                    eval_vllm_log('run.dataset_build_start', dataset_name=dataset_name)
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    eval_vllm_log(
                        'run.dataset_build_done',
                        dataset_name=dataset_name,
                        modality=getattr(dataset, 'MODALITY', None),
                        dataset_type=getattr(dataset, 'TYPE', None),
                        rows=len(getattr(dataset, 'data', [])) if hasattr(dataset, 'data') else None,
                    )
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            eval_vllm_log('run.dataset_build_rank0_prepare_start', dataset_name=dataset_name)
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                            eval_vllm_log('run.dataset_build_rank0_prepare_done', dataset_name=dataset_name)
                        traced_barrier(f'after_rank0_dataset_prepare:{dataset_name}')

                    eval_vllm_log('run.dataset_build_start', dataset_name=dataset_name)
                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    eval_vllm_log(
                        'run.dataset_build_done',
                        dataset_name=dataset_name,
                        modality=getattr(dataset, 'MODALITY', None),
                        dataset_type=getattr(dataset, 'TYPE', None),
                        rows=len(getattr(dataset, 'data', [])) if hasattr(dataset, 'data') else None,
                    )
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue

                # Handling Multi-Turn Dataset
                result_file = osp.join(pred_root, result_file_base)
                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(
                        pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                        dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                    )

                traced_barrier(f'after_reuse_prepare:{model_name}:{dataset_name}')

                if model is None:
                    model = model_name  # which is only a name

                if args.mode != "eval":
                    # Perform the Inference
                    if dataset.MODALITY == 'VIDEO':
                        model = infer_data_job_video(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            result_file_name=result_file_base,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            use_vllm=args.use_vllm)
                    elif dataset.TYPE == 'MT':
                        model = infer_data_job_mt(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            ignore_failed=args.ignore,
                            use_vllm=args.use_vllm)
                    else:
                        model = infer_data_job(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            ignore_failed=args.ignore,
                            use_vllm=args.use_vllm)

                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    print(dataset_name)
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
                        ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
                    ):
                        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-mini'
                        elif listinstr(['VisualPuzzles'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        elif listinstr(['PuzzleVQA'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        elif listinstr(['VisuLogic'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        else:
                            judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                        if listinstr(['LLaVABench_KO'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-0806'
                        else:
                            judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(['VGRPBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'LENS', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning', 'VTCBench', 'Asclepius'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['OlympiadBench'], dataset_name):
                        use_api_judger = judge_kwargs.get("olympiad_use_api_judger", False)
                        if use_api_judger:
                            judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['ChartMimic'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['VDC'], dataset_name):
                        judge_kwargs['model'] = 'llama31-8b'
                    elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
                        judge_kwargs['model'] = 'qwen-72b'
                    elif listinstr(['MMVMBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1'
                    elif listinstr(['M4Bench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['AyaVisionBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1'
                    elif listinstr(['MathCanvas'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1-2025-04-14'
                    elif listinstr(['MMReason'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1',
                    elif listinstr(['CoreCognition'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4.1'
                    elif listinstr(['WorldVQA'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o-1120'

                if args.use_verifier:
                    judge_kwargs['use_verifier'] = True
                if args.use_vllm:
                    judge_kwargs['use_vllm'] = True

                if RANK == 0:
                    logger.info(judge_kwargs)

                traced_barrier(f'before_eval:{model_name}:{dataset_name}')

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                eval_vllm_log_exception(
                    'run.dataset_failed',
                    e,
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
                configured_redi = use_config and cfg['model'].get(model_name, {}).get('frame_sampling_strategy') == 'redi'
                if configured_redi or getattr(model, 'frame_sampling_strategy', None) == 'redi':
                    logger.exception(f'Model {model_name} x Dataset {dataset_name} failed under REDI: {e}')
                    raise
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue

    if WORLD_SIZE > 1:
        ensure_dist_initialized()
        traced_barrier('before_destroy_process_group')
        dist.destroy_process_group()
        eval_vllm_log('run.dist_destroyed')


if __name__ == '__main__':
    load_env()
    main()
