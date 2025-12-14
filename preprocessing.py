import os
import numpy as np
import pandas as pd
from tqdm import tqdm

base_bp = np.array(['ear_left','ear_right', 'lateral_left','lateral_right','neck','nose','tail_base']).astype('object')
import warnings
warnings.filterwarnings('ignore')


def preprocess(track, annot):
    mice = np.unique(track.mouse_id)
    avail_bp = np.unique(track.bodypart)

    if not('lateral_left' in np.unique(track.bodypart)): 
        track.bodypart = track.bodypart.replace({'hip_left':'lateral_left', 'hip_right':'lateral_right'})

    # Annotation
    if isinstance(annot,pd.DataFrame):
        annot['video_frame'] = annot[['start_frame', 'stop_frame']].apply(lambda x: np.arange(*x), axis=1)
        annot = annot.explode('video_frame').drop(columns=['start_frame', 'stop_frame'])

        mice = np.unique(np.concatenate([np.unique(annot.agent_id), np.unique(annot.target_id)]))
        full = (
            pd.DataFrame({'video_frame': np.arange(track.video_frame.max()), '_': 1}).assign(_=1)
                .merge(pd.DataFrame({'agent_id': mice, '_': 1}), on='_')
                .merge(pd.DataFrame({'target_id': mice, '_': 1}), on='_').drop(columns='_')
        )
        annot = full.merge(annot, on=['video_frame', 'agent_id', 'target_id'], how='left')
        annot['action'] = annot['action'].fillna('none')
        annot = annot.rename(columns={'agent_id':'mouse_id'})
    
    # Tracking
    full = (
        pd.DataFrame({'video_frame': np.arange(track.video_frame.max()), '_': 1}).assign(_=1)
            .merge(pd.DataFrame({'mouse_id': np.unique(track.mouse_id), '_': 1}), on='_')
            .merge(pd.DataFrame({'bodypart': list(set(track.bodypart)|set(base_bp)), '_': 1}), on='_').drop(columns='_')
    )
    track = full.merge(track, on=['video_frame', 'mouse_id', 'bodypart'], how='left')

    track['coord'] = [(None if pd.isna(x[0]) else x) for x in track[['x', 'y']].values.astype(float) ]


    track['mouse_id'] = track['mouse_id'].astype('category')
    track['bodypart'] = pd.Categorical(track['bodypart'], categories=np.unique(track['bodypart']), ordered=True)

    
    tmp = (
        track
        .set_index(['video_frame', 'mouse_id', 'bodypart'])['coord']
        .unstack('bodypart')
        .unstack('mouse_id')
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=[0, 1], sort_remaining=False)
    )
    tmp.columns = [f'{bp} - {mid}' for mid,bp in tmp.columns]
    track = tmp.reset_index(level=0)

    
    for c in track.columns: 
        idx = track[c].isna()
        track.loc[idx, c] = None

    return track, annot, mice

