"""Join F media metadata to B IDs using reported SHA256, never camera alone."""
def align_manifest(b_manifest, f_manifests, frames):
    digest = b_manifest.get('file_sha256')
    matches = [m for m in f_manifests if digest and m.get('file_sha256') == digest]
    if len(matches) != 1:
        raise ValueError('Expected exactly one F media hash match')
    f = matches[0]
    for key in ('width', 'height', 'source_offset_ms'):
        if b_manifest[key] != f[key]:
            raise ValueError('B/F metadata mismatch: '+key)
    for row in frames:
        if row['video_id'] != b_manifest['video_id']:
            raise ValueError('Unexpected B video ID')
        if not f['available_start_ms'] <= row['timestamp_ms'] < f['available_end_ms']:
            raise ValueError('B frame outside F interval')
        if not 0 <= row['frame_index'] < f['frame_count']:
            raise ValueError('B frame outside F frame count')
    # Keep B's video_id and full track_uid intact; expose the F identifier explicitly.
    return dict(f, video_id=b_manifest['video_id'], f_video_id=f['video_id'],
                metadata_source='F_git_manifest', original_hash_verified_locally=False)


def image_exclusions(queries, manifest):
    result = []
    for query in queries:
        if query['modality'] != 'image':
            continue
        source, exclusion = query['source'], query['exclusion']
        if source['video_id'] != manifest['f_video_id']:
            continue
        if exclusion['video_id'] != source['video_id'] or query['split'] != manifest['split']:
            raise ValueError('F image query video/split mismatch')
        if set(exclusion['applies_to']) != {'scene','object'} or not exclusion['all_policies']:
            raise ValueError('Unsupported exclusion scope')
        a,b = exclusion['start_ms'],exclusion['end_ms']
        if not 0 <= a < b <= manifest['duration_ms']:
            raise ValueError('Invalid exclusion interval')
        result.append(dict(video_id=manifest['video_id'], start_ms=a, end_ms=b,
                           query_id=query['query_id'], f_video_id=manifest['f_video_id']))
    return result
