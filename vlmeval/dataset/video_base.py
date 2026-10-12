from abc import abstractmethod
from ..smp import *


class VideoBaseDataset:

    MODALITY = 'VIDEO'

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False,
                 nframe=0,
                 fps=-1):
        try:
            import decord
        except Exception as e:
            logging.critical(f'{type(e)}: {e}')
            logging.critical('Please install decord via `pip install decord`.')

        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')

    @staticmethod
    def _cv2_extract_frames(src_path, indices, sparse_gap=256):
        """Fork-safe frame extraction using cv2 only (no decord global FFmpeg state).

        - Sparse sampling (large avg gap): seek per frame via CAP_PROP_POS_FRAMES.
        - Dense sampling (small avg gap): sequential grab + selective retrieve from
          the first wanted frame. Missing frames are padded with the previous frame
          so the returned count matches len(indices).
        Returns a list of BGR numpy frames.
        """
        import cv2
        indices = sorted(set(int(i) for i in indices))
        if not indices:
            return []
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {src_path}')
        avg_gap = (indices[-1] - indices[0]) / max(1, len(indices) - 1)
        frames = []
        try:
            if avg_gap > sparse_gap:
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok:
                        frames.append(frame)
                    elif frames:
                        frames.append(frames[-1])
            else:
                want = set(indices)
                start, max_idx = indices[0], indices[-1]
                if start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                pos = start
                while pos <= max_idx:
                    if not cap.grab():
                        break
                    if pos in want:
                        ok2, frame = cap.retrieve()
                        if ok2:
                            frames.append(frame)
                        elif frames:
                            frames.append(frames[-1])
                    pos += 1
        finally:
            cap.release()
        # pad to expected count when the decoder bailed early
        while len(frames) < len(indices) and frames:
            frames.append(frames[-1])
        return frames

    @staticmethod
    def _pyav_extract_frames(src_path, indices, src_fps, sparse_gap=256):
        """Universal fork-safe frame extraction via PyAV (bundled FFmpeg).

        Handles codecs the system cv2/decord builds cannot decode (e.g. AV1).
        Indices are mapped to timestamps using src_fps so the sampling matches
        the cv2 path. Returns a list of BGR numpy frames.
        """
        import av
        indices = sorted(set(int(i) for i in indices))
        if not indices:
            return []
        container = av.open(src_path)
        try:
            stream = container.streams.video[0]
            tb = stream.time_base
            times = [idx / src_fps for idx in indices]
            avg_gap = (indices[-1] - indices[0]) / max(1, len(indices) - 1)
            frames = []
            if avg_gap > sparse_gap:
                for t in times:
                    container.seek(int(t / tb), stream=stream, backward=True)
                    got = None
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            continue
                        if float(frame.pts * tb) >= t - 1e-3:
                            got = frame
                            break
                    if got is not None:
                        frames.append(got.to_ndarray(format='bgr24'))
                    elif frames:
                        frames.append(frames[-1])
            else:
                ti = 0
                container.seek(int(times[0] / tb), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    ft = float(frame.pts * tb)
                    while ti < len(times) and ft >= times[ti] - 1e-3:
                        frames.append(frame.to_ndarray(format='bgr24'))
                        ti += 1
                    if ti >= len(times):
                        break
        finally:
            container.close()
        while len(frames) < len(indices) and frames:
            frames.append(frames[-1])
        return frames

    @classmethod
    def _extract_video_frames(cls, src_path, indices, src_fps):
        """Extract frames with cv2 (fast path); fall back to PyAV when cv2 cannot
        decode the codec (e.g. AV1 on a cv2 build without dav1d). Both paths are
        multiprocessing/fork safe. Returns a list of BGR numpy frames."""
        n_expected = len(set(int(i) for i in indices))
        frames = []
        try:
            frames = cls._cv2_extract_frames(src_path, indices)
        except Exception:
            frames = []
        if len(frames) >= n_expected and n_expected > 0:
            return frames
        return cls._pyav_extract_frames(src_path, indices, src_fps)

    @staticmethod
    def _write_video_h264(out_path, frames_bgr, fps):
        """Write BGR numpy frames to an H.264 mp4 using PyAV.

        Torchvision (used by qwen_vl_utils) cannot parse mp4v-encoded files
        reliably — it returns metadata without video_fps. H.264 works fine.
        frames_bgr: list of HxWx3 BGR uint8 numpy arrays.
        """
        import av
        import numpy as np
        if not frames_bgr:
            raise ValueError('No frames to write')
        # qwen_vl_utils smart_nframes requires total_frames >= FRAME_FACTOR (2).
        if len(frames_bgr) < 2:
            frames_bgr = list(frames_bgr) * 2
        H, W = frames_bgr[0].shape[:2]
        # Ensure even dimensions for yuv420p
        W_enc = W if W % 2 == 0 else W - 1
        H_enc = H if H % 2 == 0 else H - 1
        container = av.open(out_path, 'w', format='mp4')
        try:
            stream = container.add_stream('h264', rate=int(round(fps)))
            stream.width = W_enc
            stream.height = H_enc
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '23', 'preset': 'ultrafast'}
            for bgr in frames_bgr:
                rgb = bgr[:H_enc, :W_enc, ::-1]
                av_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format='rgb24')
                av_frame = av_frame.reformat(format='yuv420p')
                for pkt in stream.encode(av_frame):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        finally:
            container.close()

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            assert idx < len(self.videos)
            sub_data = self.data[self.data['video'] == self.videos[idx]]
            return sub_data
        else:
            assert idx < len(self.data)
            return dict(self.data.iloc[idx])

    def frame_paths(self, video):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def frame_paths_fps(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video):
        import decord
        if self.fps > 0:
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)

            # 计算视频的总帧数和总时长
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps

            # 计算需要提取的总帧数
            required_frames = int(total_duration * self.fps)

            # 计算提取帧的间隔
            step_size = video_fps / self.fps

            # 计算提取帧的索引
            indices = [int(i * step_size) for i in range(required_frames)]

            # 提取帧并保存
            frame_paths = self.frame_paths_fps(video, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths

            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

        else:
            frame_paths = self.frame_paths(video)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                vid_path = osp.join(self.data_root, video + '.mp4')
                vid = decord.VideoReader(vid_path)
                step_size = len(vid) / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME', 'MVBench', 'MVBench_MP4',
                'LongVideoBench', 'WorldSense', 'VDC', 'MovieChat1k',
                'RTV-Bench', 'StreamingBench']

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = os.path.join(data_root, file_name_legacy)
        data_path = os.path.join(data_root, file_name)

        self.data_path = data_path
        if os.path.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if os.path.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
                download_file(url, data_path)
                update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not os.path.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
