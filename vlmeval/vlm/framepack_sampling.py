"""Shared validation for FramePack frame selection."""


def require_redi_frames(before, after, dataset=None):
    """Fail when REDI left an input video unselected because offline data is missing."""
    if len(before) != len(after):
        raise RuntimeError(f"REDI changed the message length for dataset={dataset!r}.")
    for source, selected in zip(before, after):
        if not isinstance(source, dict) or source.get('type') != 'video':
            continue
        video_path = source.get('value')
        selected_frames = selected.get('value') if isinstance(selected, dict) else None
        if isinstance(video_path, str) and (not isinstance(selected_frames, list) or not selected_frames):
            raise RuntimeError(
                f"REDI could not select frames for video={video_path!r}, dataset={dataset!r}. "
                "Check the offline CLIP scores (CLIP_SCORES), frame features "
                "(FRAME_FEATURES), and their matching video paths/frame indices."
            )
    return after
