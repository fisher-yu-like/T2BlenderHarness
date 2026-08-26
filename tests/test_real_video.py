from PIL import Image


def test_png_sequence_is_encoded_to_playable_mp4(tmp_path):
    from videoact.real_video import assemble_mp4_from_pngs

    frames = []
    for index in range(3):
        path = tmp_path / f"frame_{index:06d}.png"
        Image.new("RGB", (16, 16), (index * 40, 20, 80)).save(path)
        frames.append(path)

    output = tmp_path / "proxy.mp4"
    result = assemble_mp4_from_pngs(frames, output, fps=3)

    assert result.frame_count == 3
    assert result.width == 16
    assert result.height == 16
    assert output.exists()
    assert output.stat().st_size > 100
