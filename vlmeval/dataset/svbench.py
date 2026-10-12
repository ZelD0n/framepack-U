from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from ..local_paths import benchmark_path
from .video_base import VideoBaseDataset
import warnings


class SVBench(VideoBaseDataset):

    TYPE = 'Video-VQA'
    SYS = ''

    DATASET_SPECS = {
        'SVBench_Streaming_EN': ('Streaming', 'EN'),
        'SVBench_Streaming_ZH': ('Streaming', 'ZH'),
        'SVBench_Dialogue_EN': ('Dialogue', 'EN'),
        'SVBench_Dialogue_ZH': ('Dialogue', 'ZH'),
    }

    FRAMES_TMPL_EN = (
        "These are frames sampled from a streaming video segment in chronological order. "
        "Answer the question concisely in one sentence. Do not mention frame numbers or image sources."
    )

    FRAMES_TMPL_ZH = (
        "这些图像是从流式视频片段中按时间顺序采样的帧。请用一句话简洁回答问题，"
        "不要提到帧编号或图像来源。"
    )

    VIDEO_TMPL_EN = (
        "This is a streaming video segment. Answer the question concisely in one sentence. "
        "Do not mention frame numbers or image sources."
    )

    VIDEO_TMPL_ZH = (
        "这是一段流式视频片段。请用一句话简洁回答问题，不要提到帧编号或图像来源。"
    )

    def __init__(
        self,
        dataset='SVBench_Dialogue_EN',
        nframe=0,
        fps=-1,
        split='test',
        stream_context='segment',
        allow_missing_videos=False,
    ):
        self.split = split
        self.stream_context = stream_context
        self.allow_missing_videos = allow_missing_videos
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SPECS)

    @staticmethod
    def _svbench_root():
        return benchmark_path('SVBench')

    @classmethod
    def _parse_dataset_name(cls, dataset_name):
        if dataset_name not in cls.DATASET_SPECS:
            raise ValueError(f'Unsupported SVBench dataset: {dataset_name}')
        return cls.DATASET_SPECS[dataset_name]

    @staticmethod
    def _video_path(root, video_name, meta_video_path=None):
        candidates = []
        if meta_video_path:
            rel = meta_video_path
            if rel.startswith('SVBench/'):
                rel = rel[len('SVBench/'):]
            candidates.append(osp.join(root, rel))
        candidates.extend([
            osp.join(root, 'Video', video_name + '.mp4'),
            osp.join(root, 'Video', video_name + '.mkv'),
        ])
        for pth in candidates:
            if osp.exists(pth):
                return osp.relpath(pth, root)
        return osp.relpath(candidates[0], root)

    @staticmethod
    def _metadata(root, lang, split):
        meta_dir = osp.join(root, 'Meta', f'Meta_{lang}')
        if split == 'all':
            files = [osp.join(meta_dir, 'meta_train.csv'), osp.join(meta_dir, 'meta_test.csv')]
            frames = [pd.read_csv(pth) for pth in files if osp.exists(pth)]
            if frames:
                return pd.concat(frames, ignore_index=True)
        else:
            meta_file = osp.join(meta_dir, f'meta_{split}.csv')
            if osp.exists(meta_file):
                return pd.read_csv(meta_file)
        raise FileNotFoundError(f'SVBench metadata is not found under {meta_dir} for split={split!r}')

    @staticmethod
    def _source_path(root, rel_path):
        rel = rel_path
        if rel.startswith('SVBench/'):
            rel = rel[len('SVBench/'):]
        return osp.join(root, rel)

    def _should_skip_missing_video(self, root, video_path, video_name):
        abs_video_path = osp.join(root, video_path)
        if osp.exists(abs_video_path):
            return False
        msg = f'SVBench video is missing for {video_name}: {abs_video_path}'
        if self.allow_missing_videos:
            warnings.warn(f'{msg}; skipping because allow_missing_videos=True.')
            return True
        raise FileNotFoundError(f'{msg}. Set allow_missing_videos=True to skip missing videos explicitly.')

    def _streaming_rows(self, root, meta, lang):
        rows = []
        for _, meta_row in meta.iterrows():
            video_name = meta_row['Video_Name']
            ann_path = self._source_path(root, meta_row['Path_of_StreamingPathData'])
            records = load(ann_path)
            video_path = self._video_path(root, video_name, meta_row.get('Path_of_Video'))
            if self._should_skip_missing_video(root, video_path, video_name):
                continue
            for path_id, path_data in enumerate(records):
                questions = path_data.get('questions', [])
                answers = path_data.get('answers', [])
                timestamps = path_data.get('timestamps', [])
                for turn_id, question in enumerate(questions):
                    start_time, end_time = timestamps[turn_id]
                    rows.append(dict(
                        video=video_name,
                        video_path=video_path,
                        task='Streaming',
                        lang=lang,
                        split=meta_row.get('Sort_of_Set', self.split),
                        source=meta_row.get('Source_of_Dataset', ''),
                        path_id=path_id,
                        chain_id=-1,
                        turn_id=turn_id,
                        start_time=float(start_time),
                        end_time=float(end_time),
                        question=question,
                        answer=answers[turn_id],
                    ))
        return rows

    def _dialogue_rows(self, root, meta, lang):
        rows = []
        for _, meta_row in meta.iterrows():
            video_name = meta_row['Video_Name']
            ann_path = self._source_path(root, meta_row['Path_of_Dialogue'])
            records = load(ann_path)
            video_path = self._video_path(root, video_name, meta_row.get('Path_of_Video'))
            if self._should_skip_missing_video(root, video_path, video_name):
                continue
            for chain_id, item in enumerate(records):
                chain = item.get('chain', {})
                questions = chain.get('questions', [])
                answers = chain.get('answers', [])
                start_time = float(item.get('qac_timestamps_start', 0))
                end_time = float(item.get('qac_timestamps_end', 0))
                for turn_id, question in enumerate(questions):
                    rows.append(dict(
                        video=video_name,
                        video_path=video_path,
                        task='Dialogue',
                        lang=lang,
                        split=meta_row.get('Sort_of_Set', self.split),
                        source=meta_row.get('Source_of_Dataset', ''),
                        path_id=-1,
                        chain_id=chain_id,
                        turn_id=turn_id,
                        start_time=start_time,
                        end_time=end_time,
                        question=question,
                        answer=answers[turn_id],
                    ))
        return rows

    def prepare_dataset(self, dataset_name='SVBench_Dialogue_EN', repo_id='yzy666/SVBench'):
        root = self._svbench_root()
        task, lang = self._parse_dataset_name(dataset_name)
        data_file = osp.join(root, f'{dataset_name}_{self.split}.tsv')

        if not osp.exists(osp.join(root, 'Video')):
            raise FileNotFoundError(f'SVBench videos are not found under {root}/Video')

        def generate_tsv():
            meta = self._metadata(root, lang, self.split)
            if task == 'Streaming':
                rows = self._streaming_rows(root, meta, lang)
            elif task == 'Dialogue':
                rows = self._dialogue_rows(root, meta, lang)
            else:
                raise ValueError(f'Unsupported SVBench task: {task}')

            data = pd.DataFrame(rows)
            data.insert(0, 'index', np.arange(len(data)))
            cols = [
                'index', 'video', 'video_path', 'task', 'lang', 'split', 'source',
                'path_id', 'chain_id', 'turn_id', 'start_time', 'end_time', 'question', 'answer'
            ]
            data = data[cols]
            data.to_csv(data_file, sep='\t', index=False)

        if not osp.exists(data_file) or os.environ.get('FORCE_REBUILD_SVBENCH', None):
            generate_tsv()

        return dict(data_file=data_file, root=root)

    def _available_frame_indices(self, vid, start_time, end_time):
        fps = vid.get_avg_fps()
        n_frames = len(vid)
        if self.stream_context == 'history':
            start_time = 0.0
        start_frame = max(0, min(n_frames - 1, int(float(start_time) * fps)))
        end_frame = max(start_frame, min(n_frames - 1, int(float(end_time) * fps)))
        span = end_frame - start_frame + 1

        if self.nframe > 0 and self.fps < 0:
            count = min(self.nframe, span)
            if count <= 1:
                return [start_frame], fps, n_frames
            indices = np.linspace(start_frame, end_frame, count, dtype=int).tolist()
        elif self.fps > 0:
            step = max(1, int(round(fps / self.fps)))
            indices = list(range(start_frame, end_frame + 1, step))
            if indices[-1] != end_frame:
                indices.append(end_frame)
        else:
            raise ValueError('fps and nframe should be set at least one valid value')
        return sorted(set(indices)), fps, n_frames

    def _frame_paths_for_line(self, line, num_frames):
        stem = osp.splitext(line['video_path'])[0]
        start_ms = int(round(float(line['start_time']) * 1000))
        end_ms = int(round(float(line['end_time']) * 1000))
        prefix = 'history' if self.stream_context == 'history' else 'segment'
        clip_id = f'{stem}_{prefix}_{start_ms:010d}_{end_ms:010d}ms'
        frame_root = osp.join(self.frame_root, clip_id)
        os.makedirs(frame_root, exist_ok=True)
        if self.nframe > 0:
            return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]
        return [
            osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, self.fps))
            for i in range(1, num_frames + 1)
        ]

    def save_video_frames(self, line, video_llm=False):
        import decord

        vid_path = osp.join(self.data_root, line['video_path'])
        if not osp.exists(vid_path):
            raise FileNotFoundError(f'SVBench video is not found: {vid_path}')

        vid = decord.VideoReader(vid_path, ctx=decord.cpu(0), num_threads=1)
        indices, fps, n_frames = self._available_frame_indices(vid, line['start_time'], line['end_time'])
        video_info = {
            'fps': fps,
            'n_frames': n_frames,
            'start_time': float(line['start_time']),
            'end_time': float(line['end_time']),
            'start_frame': indices[0],
            'end_frame': indices[-1],
        }

        if video_llm:
            return [], indices, video_info

        frame_paths = self._frame_paths_for_line(line, len(indices))
        if np.all([osp.exists(p) for p in frame_paths]):
            return frame_paths, indices, video_info

        lock_path = osp.join(osp.dirname(frame_paths[0]), '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=300):
            if not np.all([osp.exists(p) for p in frame_paths]):
                images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
        return frame_paths, indices, video_info

    def _clip_video_path(self, line):
        src_path = osp.join(self.data_root, line['video_path'])
        stem = osp.splitext(line['video_path'])[0]
        start_ms = int(round(float(line['start_time']) * 1000))
        end_ms = int(round(float(line['end_time']) * 1000))
        if self.stream_context == 'history':
            start_ms = 0
        cfg_tag = f'n{self.nframe}' if self.nframe > 0 else f'fps{self.fps}'
        out_path = osp.join(self.frame_root, 'video_clips', f'{stem}_{start_ms:010d}_{end_ms:010d}ms_{cfg_tag}.mp4')
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        if osp.exists(out_path) and osp.getsize(out_path) > 0:
            return out_path

        lock_path = out_path + '.lock'
        with portalocker.Lock(lock_path, 'w', timeout=300):
            if osp.exists(out_path) and osp.getsize(out_path) > 0:
                return out_path
            import cv2

            # Probe source via cv2 (fork-safe; decord global FFmpeg state is not).
            cap = cv2.VideoCapture(src_path)
            if not cap.isOpened():
                raise RuntimeError(f'Failed to open SVBench video: {src_path}')
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            start_time = 0.0 if self.stream_context == 'history' else max(0.0, float(line.get('start_time', 0)))
            end_time = float(line['end_time'])
            start_frame = min(n_total - 1, max(0, int(start_time * src_fps)))
            end_frame = min(n_total - 1, max(start_frame, int(end_time * src_fps)))

            if self.nframe > 0:
                count = min(self.nframe, end_frame - start_frame + 1)
                count = max(4, count // 4 * 4)
                indices = np.linspace(start_frame, end_frame, count, dtype=int).tolist()
                out_fps = src_fps
            else:
                step = max(1, int(round(src_fps / self.fps)))
                indices = list(range(start_frame, end_frame + 1, step))[:768]
                if not indices:
                    indices = [start_frame]
                out_fps = self.fps

            frames = self._extract_video_frames(src_path, indices, src_fps)
            if not frames:
                raise RuntimeError(f'Failed to extract frames from SVBench video: {src_path}')
            tmp_path = out_path + '.tmp'
            try:
                self._write_video_h264(tmp_path, frames, out_fps)
                os.rename(tmp_path, out_path)
            finally:
                if osp.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            if not osp.exists(out_path) or osp.getsize(out_path) == 0:
                raise RuntimeError(f'Failed to write SVBench clipped video: {out_path}')
        return out_path

    @staticmethod
    def _format_question(line):
        return f"Question: {line['question']}\nAnswer: "

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        line = line.copy()

        is_zh = str(line['lang']).upper() == 'ZH'
        frames_tmpl = self.FRAMES_TMPL_ZH if is_zh else self.FRAMES_TMPL_EN
        video_tmpl = self.VIDEO_TMPL_ZH if is_zh else self.VIDEO_TMPL_EN

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            seg_start = 0.0 if self.stream_context == 'history' else max(0.0, float(line.get('start_time', 0) or 0))
            # Source + effective segment for offline frame selection (see RTVBench).
            message.append(dict(
                type='video',
                value=self._clip_video_path(line),
                _source_video_path=osp.join(self.data_root, line['video_path']),
                _segment_times=[seg_start, float(line['end_time'])],
            ))
            message.append(dict(type='text', value=video_tmpl))
        else:
            frames, _, _ = self.save_video_frames(line, video_llm=False)
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=frames_tmpl))
        message.append(dict(type='text', value=self._format_question(line)))
        return message

    @staticmethod
    def _normalize_text(text):
        import re
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]
            for idx in data['index']:
                ans = cls._normalize_text(data.loc[data['index'] == idx, 'answer'].values[0])
                pred = cls._normalize_text(data.loc[data['index'] == idx, 'prediction'].values[0])
                if not pred:
                    score = -1
                elif ans == pred:
                    score = 1
                elif ans in pred or pred in ans:
                    score = 0.5
                else:
                    score = 0
                data.loc[data['index'] == idx, 'score'] = score
            print(
                f'Among {len(data)} questions, failed to obtain prediction for '
                f'{len(data) - len(data_un)} questions. The built-in SVBench score is a simple text match; '
                'use the official SVBench submission format for leaderboard evaluation.'
            )
            dump(data, score_file)

        data = load(score_file)
        valid = data[data['score'] >= 0]
        rating = {
            'overall': f"{valid['score'].mean():.3f}" if len(valid) else 'nan',
            'task': {},
            'source': {},
        }
        for task in sorted(data['task'].unique()):
            sub = data[(data['task'] == task) & (data['score'] >= 0)]
            rating['task'][task] = f"{sub['score'].mean():.3f}" if len(sub) else 'nan'
        for source in sorted(data['source'].unique()):
            sub = data[(data['source'] == source) & (data['score'] >= 0)]
            rating['source'][source] = f"{sub['score'].mean():.3f}" if len(sub) else 'nan'
        dump(rating, rating_file)
        return rating
