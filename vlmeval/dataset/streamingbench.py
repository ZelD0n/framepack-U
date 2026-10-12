from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from ..local_paths import benchmark_path
from .video_base import VideoBaseDataset


class StreamingBench(VideoBaseDataset):

    TYPE = 'Video-MCQ'
    SYS = ''

    SUBSETS = {
        'StreamingBench_Real_Time_Visual_Understanding': ['Real_Time_Visual_Understanding.csv'],
        'StreamingBench_Sequential_Question_Answering': ['Sequential_Question_Answering.csv'],
        'StreamingBench_Contextual_Understanding': ['Contextual_Understanding.csv'],
        'StreamingBench_Omni_Source_Understanding': ['Omni_Source_Understanding.csv'],
        'StreamingBench_Proactive_Output': ['Proactive_Output.csv'],
        'StreamingBench': [
            'Real_Time_Visual_Understanding.csv',
            'Sequential_Question_Answering.csv',
            'Contextual_Understanding.csv',
            'Omni_Source_Understanding.csv',
            'Proactive_Output.csv',
        ],
    }

    FRAMES_TMPL = (
        "These are frames sampled from a streaming video in chronological order. The frames only cover video "
        "content available up to the current question timestamp. Answer based only on the visible video so far. "
        "For multiple-choice questions, respond with only the letter of the correct option."
    )

    VIDEO_TMPL = (
        "This is a streaming video segment cut at the current question timestamp. Answer based only on the "
        "visible video so far. For multiple-choice questions, respond with only the letter of the correct option."
    )

    PROACTIVE_FRAMES_TMPL = (
        "These are frames sampled from a streaming video in chronological order. Follow the instruction and "
        "respond with only the requested output text."
    )

    PROACTIVE_VIDEO_TMPL = (
        "This is a streaming video segment. Follow the instruction and respond with only the requested output text."
    )

    def __init__(self, dataset='StreamingBench', nframe=0, fps=-1, stream_context='history'):
        self.stream_context = stream_context
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return list(cls.SUBSETS)

    @staticmethod
    def _streamingbench_root():
        return benchmark_path('StreamingBench')

    @staticmethod
    def _parse_timestamp(value):
        if pd.isna(value):
            return 0.0
        text = str(value).strip()
        if not text:
            return 0.0
        parts = text.split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(text)
        except ValueError as e:
            raise ValueError(f'Invalid StreamingBench timestamp: {value!r}') from e

    @staticmethod
    def _sample_id(question_id):
        import re
        match = re.search(r'_sample_(\d+)(?:_|$)', str(question_id))
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _normalize_options(options):
        if isinstance(options, (list, tuple)):
            return list(options)
        if pd.isna(options):
            return []
        if isinstance(options, str):
            try:
                import ast
                options = ast.literal_eval(options)
            except Exception:
                options = [x.strip() for x in options.split('\n') if x.strip()]
        return list(options)

    @staticmethod
    def _range_archive(prefix, sample_id, chunk_size=50):
        start = ((sample_id - 1) // chunk_size) * chunk_size + 1
        end = start + chunk_size - 1
        return f'{prefix}_{start}-{end}'

    @classmethod
    def _archive_name(cls, subset, task_type, sample_id):
        if subset == 'Real_Time_Visual_Understanding':
            return cls._range_archive('Real-Time Visual Understanding', sample_id)
        if subset == 'Sequential_Question_Answering':
            return cls._range_archive('Sequential Question Answering', sample_id, chunk_size=25)
        if subset == 'Contextual_Understanding':
            task_map = {
                'Anomaly Context Understanding': 'Anomaly Context Understanding',
                'Misleading Context Recognition': 'Misleading Context Understanding',
            }
            return task_map.get(str(task_type), str(task_type))
        if subset == 'Omni_Source_Understanding':
            if task_type == 'Scene Understanding':
                return cls._range_archive('Scene Understanding', sample_id, chunk_size=25)
            task_map = {
                'Emotion Recognition': 'Emotion Recognition',
                'Multimodal Alignment': 'Multimodal Alignment',
                'Source Discrimination': 'Source Discrimination',
            }
            return task_map.get(str(task_type), str(task_type))
        if subset == 'Proactive_Output':
            return cls._range_archive('Proactive Output', sample_id, chunk_size=25)
        return subset

    def prepare_dataset(self, dataset_name='StreamingBench', repo_id='mjuicem/StreamingBench'):
        root = self._streamingbench_root()
        meta_dir = osp.join(root, 'StreamingBench')
        data_file = osp.join(root, f'{dataset_name}.tsv')

        if dataset_name not in self.SUBSETS:
            raise ValueError(f'Unsupported StreamingBench dataset: {dataset_name}')
        if not osp.exists(meta_dir):
            raise FileNotFoundError(
                f'StreamingBench metadata is not found: {meta_dir}. '
                'Link the prepared dataset with scripts/setup_local_paths.py.'
            )

        def generate_tsv():
            rows = []
            for csv_name in self.SUBSETS[dataset_name]:
                csv_path = osp.join(meta_dir, csv_name)
                if not osp.exists(csv_path):
                    raise FileNotFoundError(f'StreamingBench CSV is not found: {csv_path}')
                data = pd.read_csv(csv_path)
                subset = osp.splitext(csv_name)[0]
                for _, item in data.iterrows():
                    item = item.to_dict()
                    sample_id = self._sample_id(item.get('question_id'))
                    if sample_id is None:
                        raise ValueError(
                            f'Cannot infer sample id from StreamingBench question_id={item.get("question_id")!r} '
                            f'in {csv_path}.'
                        )

                    is_proactive = subset == 'Proactive_Output'
                    end_raw = item.get('ground_truth_time_stamp') if is_proactive else item.get('time_stamp')
                    end_time = self._parse_timestamp(end_raw)
                    start_time = 0.0 if self.stream_context == 'history' else self._parse_timestamp(item.get('time_stamp'))
                    if is_proactive and self.stream_context != 'history':
                        start_time = self._parse_timestamp(item.get('time_stamp'))

                    archive = self._archive_name(subset, item.get('task_type', ''), sample_id)
                    video_path = osp.join('videos_by_archive', archive, f'sample_{sample_id}', 'video.mp4')
                    abs_video_path = osp.join(root, video_path)
                    if not osp.exists(abs_video_path):
                        fallback_video_path = osp.join(f'sample_{sample_id}', 'video.mp4')
                        fallback_abs_video_path = osp.join(root, fallback_video_path)
                        if osp.exists(fallback_abs_video_path):
                            video_path = fallback_video_path
                            abs_video_path = fallback_abs_video_path
                    if not osp.exists(abs_video_path):
                        raise FileNotFoundError(f'StreamingBench video is not found: {abs_video_path}')

                    rows.append(dict(
                        video=f'sample_{sample_id}',
                        video_path=video_path,
                        archive=archive,
                        subset=subset,
                        task_type=item.get('task_type', ''),
                        question_id=item.get('question_id', ''),
                        start_time=float(start_time),
                        end_time=float(end_time),
                        time_stamp=item.get('time_stamp', ''),
                        ground_truth_time_stamp=item.get('ground_truth_time_stamp', ''),
                        question=item.get('question', ''),
                        options=self._normalize_options(item.get('options')),
                        answer=item.get('ground_truth_output') if is_proactive else item.get('answer'),
                        answer_type='text' if is_proactive else 'mcq',
                        frames_required=item.get('frames_required', ''),
                        temporal_clue_type=item.get('temporal_clue_type', ''),
                    ))

            data = pd.DataFrame(rows)
            data.insert(0, 'index', np.arange(len(data)))
            cols = [
                'index', 'video', 'video_path', 'subset', 'task_type', 'question_id',
                'archive', 'start_time', 'end_time', 'time_stamp', 'ground_truth_time_stamp',
                'question', 'options', 'answer', 'answer_type', 'frames_required', 'temporal_clue_type'
            ]
            data = data[cols]
            data.to_csv(data_file, sep='\t', index=False)

        if not osp.exists(data_file) or os.environ.get('FORCE_REBUILD_STREAMINGBENCH', None):
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
            raise FileNotFoundError(f'StreamingBench video is not found: {vid_path}')

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
                raise RuntimeError(f'Failed to open StreamingBench video: {src_path}')
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
                raise RuntimeError(f'Failed to extract frames from StreamingBench video: {src_path}')
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
                raise RuntimeError(f'Failed to write StreamingBench clipped video: {out_path}')
        return out_path

    @staticmethod
    def _format_question(line):
        if str(line.get('answer_type', 'mcq')) == 'text':
            return f"Instruction: {line['question']}\nOutput: "
        options = StreamingBench._normalize_options(line['options'])
        if options and not str(options[0]).strip().startswith('A.'):
            labels = 'ABCD'
            options = [f'{labels[i]}. {opt}' for i, opt in enumerate(options[:4])]
        options_str = '\n'.join(options)
        return f'Question: {line["question"]}\n\nOptions:\n{options_str}\n\nThe best option is:'

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        line = line.copy()

        is_text = str(line.get('answer_type', 'mcq')) == 'text'
        frames_tmpl = self.PROACTIVE_FRAMES_TMPL if is_text else self.FRAMES_TMPL
        video_tmpl = self.PROACTIVE_VIDEO_TMPL if is_text else self.VIDEO_TMPL

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
    def _extract_mcq_answer(pred):
        import re
        s = str(pred).strip()
        # word-boundary match first (most precise)
        m = re.search(r'\b([ABCD])\b', s.upper())
        if m:
            return m.group(1)
        # prefix keyword pattern
        m = re.search(r'(?:OPTION|ANSWER)\s*[:\-]?\s*([ABCD])', s.upper())
        if m:
            return m.group(1)
        # first character fallback
        c = s[:1].upper()
        if c in 'ABCD':
            return c
        return ''

    @staticmethod
    def _normalize_text(text):
        import re
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _mean_score(values):
        vals = [x for x in values if x >= 0]
        return f'{np.mean(vals):.3f}' if len(vals) else 'nan'

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
                row = data[data['index'] == idx].iloc[0]
                pred = row.get('prediction', '')
                if pd.isna(pred) or str(pred).strip() == '':
                    score = -1
                elif row.get('answer_type', 'mcq') == 'text':
                    ans = cls._normalize_text(row['answer'])
                    pred_norm = cls._normalize_text(pred)
                    if ans == pred_norm:
                        score = 1
                    elif ans in pred_norm or pred_norm in ans:
                        score = 0.5
                    else:
                        score = 0
                else:
                    extracted = cls._extract_mcq_answer(pred)
                    score = int(extracted == row['answer']) if extracted else -1
                data.loc[data['index'] == idx, 'score'] = score

            rejected = [x for x in data['score'] if x == -1]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                'Proactive Output uses a simple text-match proxy; use the official StreamingBench evaluator for '
                'leaderboard-grade timing metrics.'
            )
            dump(data, score_file)

        data = load(score_file)
        rating = {
            'overall': cls._mean_score(data['score'].tolist()),
            'subset': {},
            'task_type': {},
            'temporal_clue_type': {},
            'frames_required': {},
        }
        for key in ['subset', 'task_type', 'temporal_clue_type', 'frames_required']:
            for val in sorted(data[key].dropna().unique()):
                sub = data[data[key] == val]
                rating[key][val] = cls._mean_score(sub['score'].tolist())
        dump(rating, rating_file)
        return rating
