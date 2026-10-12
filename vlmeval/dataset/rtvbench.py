from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from ..local_paths import benchmark_path
from .video_base import VideoBaseDataset


class RTVBench(VideoBaseDataset):

    SYS = ''

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    FRAMES_TMPL = (
        "These are frames sampled from a streaming video. The frames are in chronological order and only cover "
        "the video segment available up to the current question timestamp. Select the best answer to the "
        "following multiple-choice question based only on the visible video so far. Respond with only the "
        "letter of the correct option."
    )

    VIDEO_TMPL = (
        "This is a streaming video segment cut at the current question timestamp. Select the best answer to the "
        "following multiple-choice question based only on the visible video so far. Respond with only the "
        "letter of the correct option."
    )

    FIELD_TO_DIR = {
        'driving_cam': 'driving_cam',
        'sports_cam': 'sports_cam',
        'embodied_perspective': 'embodied_perspective',
    }

    def __init__(self, dataset='RTV-Bench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['RTV-Bench']

    @staticmethod
    def _rtv_root():
        return benchmark_path('RTV-Bench')

    @staticmethod
    def _video_rel_path(field, video):
        folder = RTVBench.FIELD_TO_DIR.get(field, field)
        return osp.join('videos', folder, video)

    def prepare_dataset(self, dataset_name='RTV-Bench', repo_id='xunsh/RTV-Bench'):
        dataset_path = self._rtv_root()
        qa_file = osp.join(dataset_path, 'QA.json')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        if not osp.exists(qa_file):
            raise FileNotFoundError(
                f'RTV-Bench QA file is not found: {qa_file}. '
                'Link the prepared dataset with scripts/setup_local_paths.py.'
            )

        def generate_tsv():
            records = load(qa_file)
            rows = []
            for idx, item in enumerate(records):
                item = dict(item)
                item['index'] = idx
                item['video_path'] = self._video_rel_path(item['field'], item['video'])
                item['candidates'] = [
                    f'{key}. {value}' for key, value in sorted(item['options'].items())
                ]
                rows.append(item)
            data = pd.DataFrame(rows)
            cols = [
                'index', 'video', 'video_path', 'questionID', 'type', 'field',
                'start_time', 'end_time', 'question', 'candidates', 'answer'
            ]
            data = data[cols]
            data.to_csv(data_file, sep='\t', index=False)

        if not osp.exists(data_file) or os.environ.get('FORCE_REBUILD_RTVBENCH', None):
            generate_tsv()

        return dict(data_file=data_file, root=dataset_path)

    def _available_frame_indices(self, vid, end_time):
        fps = vid.get_avg_fps()
        n_frames = len(vid)
        end_frame = max(0, min(n_frames - 1, int(float(end_time) * fps)))
        if self.nframe > 0 and self.fps < 0:
            count = min(self.nframe, end_frame + 1)
            if count <= 1:
                return [0], fps, n_frames
            indices = np.linspace(0, end_frame, count, dtype=int).tolist()
        elif self.fps > 0:
            step = max(1, int(round(fps / self.fps)))
            indices = list(range(0, end_frame + 1, step))
            if indices[-1] != end_frame:
                indices.append(end_frame)
        else:
            raise ValueError('fps and nframe should be set at least one valid value')
        return sorted(set(indices)), fps, n_frames

    def _frame_paths_for_line(self, line, num_frames):
        stem = osp.splitext(line['video_path'])[0]
        clip_id = f"{stem}_to_{int(round(float(line['end_time']) * 1000)):010d}ms"
        frame_root = osp.join(self.frame_root, clip_id)
        os.makedirs(frame_root, exist_ok=True)
        if self.nframe > 0:
            tmpl = self.frame_tmpl
        else:
            tmpl = self.frame_tmpl_fps
        if self.nframe > 0:
            return [osp.join(frame_root, tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]
        return [osp.join(frame_root, tmpl.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, line, video_llm=False):
        import decord

        vid_path = osp.join(self.data_root, line['video_path'])
        if not osp.exists(vid_path):
            raise FileNotFoundError(f'RTV-Bench video is not found: {vid_path}')

        vid = decord.VideoReader(vid_path, ctx=decord.cpu(0), num_threads=1)
        indices, fps, n_frames = self._available_frame_indices(vid, line['end_time'])
        video_info = {
            'fps': fps,
            'n_frames': n_frames,
            'end_time': float(line['end_time']),
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
        end_ms = int(round(float(line['end_time']) * 1000))
        cfg_tag = f'n{self.nframe}' if self.nframe > 0 else f'fps{self.fps}'
        out_path = osp.join(self.frame_root, 'video_clips', f'{stem}_to_{end_ms:010d}ms_{cfg_tag}.mp4')
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
                raise RuntimeError(f'Failed to open RTV-Bench video: {src_path}')
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            end_time = float(line['end_time'])
            end_frame = min(n_total - 1, max(0, int(end_time * src_fps)))

            if self.nframe > 0:
                # nframe mode: save at source fps so model uses nframes= path
                count = min(self.nframe, end_frame + 1)
                count = max(4, count // 4 * 4)
                indices = np.linspace(0, end_frame, count, dtype=int).tolist()
                out_fps = src_fps
            else:
                # fps mode: save at target fps so model samples at correct rate.
                # Cap at 768 to match typical max_frames model config.
                step = max(1, int(round(src_fps / self.fps)))
                indices = list(range(0, end_frame + 1, step))[:768]
                if not indices:
                    indices = [0]
                out_fps = self.fps

            frames = self._extract_video_frames(src_path, indices, src_fps)
            if not frames:
                raise RuntimeError(f'Failed to extract frames from RTV-Bench video: {src_path}')
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
                raise RuntimeError(f'Failed to write RTV-Bench clipped video: {out_path}')
        return out_path

    @staticmethod
    def _format_question(line):
        candidates = eval(line['candidates']) if isinstance(line['candidates'], str) else line['candidates']
        question = line['question'] + '\n' + '\n'.join(candidates)
        return f'Question: {question}\nAnswer: '

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        line = line.copy()

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            # Attach the source video + effective segment so models that do their
            # own offline frame selection (e.g. FramePack redi/clip) can score and
            # decode against ALL native frames of the segment rather than the
            # fps-subsampled clip. The clip remains the value for plain video LLMs.
            message.append(dict(
                type='video',
                value=self._clip_video_path(line),
                _source_video_path=osp.join(self.data_root, line['video_path']),
                _segment_times=[0.0, float(line['end_time'])],
            ))
            message.append(dict(type='text', value=self.VIDEO_TMPL))
        else:
            frames, _, _ = self.save_video_frames(line, video_llm=False)
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=self.FRAMES_TMPL))
        message.append(dict(type='text', value=self._format_question(line)))
        return message

    @staticmethod
    def _parse_group_id(qid):
        import re
        m = re.match(r'^q-group-([a-z0-9]+)-([012])(?:-.+)?$', str(qid).strip(), re.IGNORECASE)
        if not m:
            return None, None
        return m.group(1), m.group(2)

    @staticmethod
    def _parse_main_category(type_str):
        for cat in ('Object', 'Action', 'Event'):
            if str(type_str).startswith(f'{cat}-'):
                return cat
        return None

    @staticmethod
    def _parse_dimension(type_str):
        s = str(type_str)
        if 'SR' in s or 'FP' in s:
            return 'Reasoning'
        if 'IA' in s or 'PU' in s or 'GU' in s:
            return 'Understanding'
        if 'TP' in s or 'VP' in s or 'SP' in s:
            return 'Perception'
        return None

    @staticmethod
    def _extract_answer(pred):
        import re
        s = str(pred).strip()
        # word-boundary match first (most precise)
        m = re.search(r'\b([ABCDE])\b', s.upper())
        if m:
            return m.group(1)
        # prefix keyword pattern
        m = re.search(r'(?:OPTION|ANSWER)\s*[:\-]?\s*([ABCDE])', s.upper())
        if m:
            return m.group(1)
        # first character fallback
        c = s[:1].upper()
        if c in 'ABCDE':
            return c
        return ''

    @staticmethod
    def _mean_score(values):
        vals = [x for x in values if x >= 0]
        return f'{np.mean(vals):.3f}' if len(vals) else 'nan'

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from collections import defaultdict

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                extracted = cls._extract_answer(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans) if extracted else 0

            n_no_pred = len(data) - len(data_un)
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {n_no_pred} questions.'
            )
            dump(data, score_file)

        data = load(score_file)

        # Group-aware conditional scoring (official compute_score.py protocol):
        # basic_pass = all q0 correct AND all q1 correct (missing = pass)
        # q2 score counts only when basic_pass; otherwise q2 score = 0
        groups = defaultdict(lambda: defaultdict(list))
        for _, row in data.iterrows():
            gid, qtype = cls._parse_group_id(row.get('questionID', ''))
            if gid and qtype:
                groups[gid][f'q{qtype}'].append(row)

        type_stats = defaultdict(lambda: {'score': 0, 'total': 0})
        total_score = 0
        total_q2 = 0

        for gid, g in groups.items():
            if not g['q2']:
                continue
            q0_pass = all(bool(r['score']) for r in g['q0']) if g['q0'] else True
            q1_pass = all(bool(r['score']) for r in g['q1']) if g['q1'] else True
            basic_pass = q0_pass and q1_pass

            full_type = g['q2'][0]['type']
            q2_score = sum(bool(r['score']) for r in g['q2']) if basic_pass else 0
            q2_total = len(g['q2'])

            total_score += q2_score
            total_q2 += q2_total
            type_stats[full_type]['score'] += q2_score
            type_stats[full_type]['total'] += q2_total

        def mean_str(s, t):
            return f'{s / t:.3f}' if t else 'nan'

        dim_agg = defaultdict(lambda: [0, 0])
        main_agg = defaultdict(lambda: [0, 0])
        for full_type, st in type_stats.items():
            d = cls._parse_dimension(full_type)
            m = cls._parse_main_category(full_type)
            if d:
                dim_agg[d][0] += st['score']
                dim_agg[d][1] += st['total']
            if m:
                main_agg[m][0] += st['score']
                main_agg[m][1] += st['total']

        rating = {
            'overall': mean_str(total_score, total_q2),
            'dim': {k: mean_str(v[0], v[1]) for k, v in sorted(dim_agg.items())},
            'main': {k: mean_str(v[0], v[1]) for k, v in sorted(main_agg.items())},
            'type': {k: mean_str(st['score'], st['total']) for k, st in sorted(type_stats.items())},
        }

        print(
            f'Group-aware Q2 scoring: {total_score}/{total_q2} groups processed, '
            f'overall = {rating["overall"]}'
        )
        dump(rating, tgt_file)
        return rating
