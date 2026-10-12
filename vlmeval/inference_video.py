import torch
import torch.distributed as dist
import os.path as osp
import time
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from vlmeval.debug_logger import eval_vllm_log, eval_vllm_log_exception

FAIL_MSG = 'Failed to obtain answer via API.'


def _message_summary(struct):
    if struct is None:
        return None
    if not isinstance(struct, list):
        return {'type': type(struct).__name__, 'repr': repr(struct)[:500]}
    counts = {}
    text_chars = 0
    videos = []
    images = []
    for item in struct:
        if not isinstance(item, dict):
            counts[type(item).__name__] = counts.get(type(item).__name__, 0) + 1
            continue
        item_type = item.get('type')
        counts[item_type] = counts.get(item_type, 0) + 1
        value = item.get('value')
        if item_type == 'text' and isinstance(value, str):
            text_chars += len(value)
        elif item_type == 'video':
            if isinstance(value, list):
                videos.append({'mode': 'frames', 'count': len(value)})
            else:
                videos.append({'mode': 'path', 'value': str(value)})
        elif item_type == 'image':
            images.append(str(value))
    return {
        'len': len(struct),
        'counts': counts,
        'text_chars': text_chars,
        'videos': videos[:4],
        'images_head': images[:4],
    }


def _sample_metadata(dataset, sample):
    try:
        line = dataset.data.iloc[sample]
    except Exception:
        return {'sample': sample}
    keys = [
        'index',
        'video_path',
        'subtitle_path',
        'duration',
        'starting_timestamp_for_subtitles',
        'SUB_DATASET',
    ]
    return {k: line[k] for k in keys if k in line}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    eval_vllm_log(
        'video_infer_shard',
        model_name=model_name,
        dataset_name=dataset_name,
        out_file=out_file,
        existing_results=len(res),
        env_rank=os.environ.get("RANK"),
        env_world_size=os.environ.get("WORLD_SIZE"),
        dist_initialized=dist.is_initialized(),
        rank_world=(rank, world_size),
        total=len(sample_indices),
        local=len(sample_indices_sub),
        head=sample_indices_sub[:5],
    )
    if np.all([idx in res for idx in sample_indices_sub]):
        eval_vllm_log(
            'video_infer_all_done',
            model_name=model_name,
            dataset_name=dataset_name,
            local=len(sample_indices_sub),
        )
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]
    eval_vllm_log(
        'video_infer_remaining',
        model_name=model_name,
        dataset_name=dataset_name,
        remaining=len(sample_indices_subrem),
        head=sample_indices_subrem[:5],
    )

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    try:
        if isinstance(model, str):
            eval_vllm_log(
                'video_model_build_start',
                model_name=model_name,
                kwargs=kwargs,
                ws_hidden=ws_bak is not None,
            )
            model = supported_VLM[model_name](**kwargs)
            eval_vllm_log('video_model_build_done', model_name=model_name)
        else:
            eval_vllm_log('video_model_reuse_existing', model_class=type(model).__name__)
    except Exception as e:
        eval_vllm_log_exception('video_model_build_failed', e, model_name=model_name)
        raise
    finally:
        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    for i, idx in tqdm(
        enumerate(sample_indices_subrem),
        total=len(sample_indices_subrem),
        desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}',
    ):
        if idx in res:
            continue
        item_t0 = time.monotonic()
        sample = sample_map[idx]
        eval_vllm_log(
            'video_item_start',
            model_name=model_name,
            dataset_name=dataset_name,
            loop_i=i,
            idx=idx,
            sample=sample,
            sample_metadata=_sample_metadata(dataset, sample),
        )
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            elif getattr(model, 'SUPPORTS_FPS_WITH_NFRAME', False):
                pass
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        eval_vllm_log(
            'video_build_prompt_start',
            model_name=model_name,
            dataset_name=dataset_name,
            loop_i=i,
            idx=idx,
            sample=sample,
            model_nframe=getattr(model, 'nframe', None),
            model_fps=getattr(model, 'fps', None),
            dataset_nframe=getattr(dataset, 'nframe', None),
            dataset_fps=getattr(dataset, 'fps', None),
            video_llm=getattr(model, 'VIDEO_LLM', False),
            custom_prompt=hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name),
        )
        prompt_t0 = time.monotonic()
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        eval_vllm_log(
            'video_build_prompt_done',
            model_name=model_name,
            dataset_name=dataset_name,
            loop_i=i,
            idx=idx,
            elapsed_s=round(time.monotonic() - prompt_t0, 3),
            struct_summary=_message_summary(struct),
        )
        if struct is None:
            eval_vllm_log('video_item_skip_empty_prompt', model_name=model_name, dataset_name=dataset_name, idx=idx)
            continue

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        eval_vllm_log(
            'video_generate_start',
            model_name=model_name,
            dataset_name=dataset_name,
            loop_i=i,
            idx=idx,
            struct_summary=_message_summary(struct),
        )
        gen_t0 = time.monotonic()
        fail_fast = os.environ.get('VIDEO_INFER_FAIL_FAST', os.environ.get('EVAL_VLLM_FAIL_FAST', '0'))
        fail_fast = str(fail_fast).lower() in {'1', 'true', 'yes', 'on'}
        if os.environ.get('SKIP_ERR', False) == '1' and not fail_fast and getattr(model, 'frame_sampling_strategy', None) != 'redi':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except Exception as err:
                eval_vllm_log_exception('video_generate_error_skipped', err, model_name=model_name, idx=idx)
                if isinstance(err, RuntimeError):
                    torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except Exception as e:
                eval_vllm_log_exception(
                    'video_generate_failed',
                    e,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    loop_i=i,
                    idx=idx,
                )
                raise
        eval_vllm_log(
            'video_generate_done',
            model_name=model_name,
            dataset_name=dataset_name,
            loop_i=i,
            idx=idx,
            elapsed_s=round(time.monotonic() - gen_t0, 3),
            response_chars=len(response) if isinstance(response, str) else None,
            item_elapsed_s=round(time.monotonic() - item_t0, 3),
        )

        if verbose:
            print(response, flush=True)

        res[idx] = response
        dump_interval = max(1, int(os.environ.get('VIDEO_INFER_DUMP_INTERVAL', '1')))
        if (i + 1) % dump_interval == 0:
            eval_vllm_log('video_dump_partial_start', model_name=model_name, dataset_name=dataset_name, count=len(res))
            dump(res, out_file)
            eval_vllm_log('video_dump_partial_done', model_name=model_name, dataset_name=dataset_name, count=len(res))

    res = {k: res[k] for k in sample_indices_sub}
    eval_vllm_log('video_dump_final_start', model_name=model_name, dataset_name=dataset_name, count=len(res))
    dump(res, out_file)
    eval_vllm_log('video_dump_final_done', model_name=model_name, dataset_name=dataset_name, count=len(res))
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    eval_vllm_log(
        'video_job_start',
        model_name=model_name,
        dataset_name=dataset_name,
        work_dir=work_dir,
        result_file=result_file,
        rank_world=(rank, world_size),
    )
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        eval_vllm_log('video_job_result_exists', model_name=model_name, dataset_name=dataset_name, result_file=result_file)
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm)

    if world_size > 1:
        eval_vllm_log('video_job_barrier_before_merge_start', model_name=model_name, dataset_name=dataset_name)
        dist.barrier()
        eval_vllm_log('video_job_barrier_before_merge_done', model_name=model_name, dataset_name=dataset_name)

    if rank == 0:
        eval_vllm_log('video_job_merge_start', model_name=model_name, dataset_name=dataset_name)
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            missing = [x for x in meta['index'] if x not in data_all]
            if missing:
                eval_vllm_log(
                    'video_job_merge_missing_predictions',
                    model_name=model_name,
                    dataset_name=dataset_name,
                    missing_count=len(missing),
                    missing_head=missing[:20],
                    predictions=len(data_all),
                    expected=len(meta),
                )
                fail_fast = os.environ.get('VIDEO_INFER_FAIL_FAST', os.environ.get('EVAL_VLLM_FAIL_FAST', '0'))
                fail_fast = str(fail_fast).lower() in {'1', 'true', 'yes', 'on'}
                if os.environ.get('SKIP_ERR', False) == '1' and not fail_fast and getattr(model, 'frame_sampling_strategy', None) != 'redi':
                    for x in missing:
                        data_all[x] = 'Failed to obtain answer: missing prediction during distributed merge'
                else:
                    raise RuntimeError(
                        f'Missing {len(missing)} predictions during distributed merge for '
                        f'{model_name}/{dataset_name}. First missing indices: {missing[:20]}. '
                        f'Got {len(data_all)} predictions, expected {len(meta)}.'
                    )
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
        eval_vllm_log(
            'video_job_merge_done',
            model_name=model_name,
            dataset_name=dataset_name,
            result_file=result_file,
            predictions=len(data_all),
        )
    return model
