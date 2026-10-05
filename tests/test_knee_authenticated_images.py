"""Independent RGB fixtures for the bytes that enter the public image readers."""
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError
import torch

from tralo.knee_end_to_end import development_images
from tralo.knee_yuval import Images


def png(mode='RGB', color=(19, 83, 147), size=(3, 2)):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format='PNG')
    return stream.getvalue()


def fixture(tmp_path, raw, role):
    path = tmp_path/'neutral.png'
    path.write_bytes(raw)
    return path, dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                      split='train' if role == 'training' else 'val', label=2)


def pixels(image):
    return torch.tensor(list(image.getdata()), dtype=torch.uint8).reshape(image.height, image.width, 3)


def read(root, row, role):
    if role == 'training':
        data = Images(root, [row])
        values, labels = data.batch([0], pixels)
        assert labels.tolist() == [2]
        return values[0]
    row = {k:v for k,v in row.items() if k != 'label'}
    return development_images(root, [row], pixels, 2)[0][0]


@pytest.mark.parametrize('role', ['training', 'development'])
@pytest.mark.parametrize('mode,color,expected', [
    ('RGB', (19, 83, 147), [19, 83, 147]),
    ('L', 91, [91, 91, 91]),
    ('RGBA', (29, 63, 101, 0), [29, 63, 101]),
])
def test_authenticated_pixels_preserve_native_rgb(tmp_path, role, mode, color, expected):
    _, row = fixture(tmp_path, png(mode, color), role)
    result = read(tmp_path, row, role)
    assert result.shape == (2, 3, 3)
    assert result.tolist() == [[expected]*3]*2


@pytest.mark.parametrize('role', ['training', 'development'])
def test_file_change_after_authenticated_read_cannot_change_decoded_pixels(tmp_path, monkeypatch, role):
    """Mutation occurs as the first file reader closes, before any later decode."""
    original = png()
    changed = png(color=(201, 7, 32))
    path, row = fixture(tmp_path, original, role)
    ordinary_open = Path.open
    reads = []

    class ReplaceOnClose(io.BytesIO):
        def close(self):
            if not self.closed:
                path.write_bytes(changed)
            super().close()

    def open_once(candidate, mode='r', *args, **kwargs):
        if candidate == path and mode == 'rb':
            reads.append(candidate)
            if len(reads) == 1:
                return ReplaceOnClose(original)
        return ordinary_open(candidate, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_once)
    result = read(tmp_path, row, role)
    assert result.tolist() == [[[19, 83, 147]]*3]*2
    assert reads == [path]
    assert path.read_bytes() == changed


@pytest.mark.parametrize('role', ['training', 'development'])
def test_changed_hash_is_refused_before_decoding(tmp_path, monkeypatch, role):
    path, row = fixture(tmp_path, png(), role)
    path.write_bytes(png(color=(201, 7, 32)))
    def forbidden(*args, **kwargs):
        pytest.fail('mismatched input reached decoder')
    monkeypatch.setattr(Image, 'open', forbidden)
    with pytest.raises(RuntimeError, match='image changed after audit'):
        read(tmp_path, row, role)


@pytest.mark.parametrize('role', ['training', 'development'])
def test_matching_digest_does_not_make_invalid_bytes_an_image(tmp_path, role):
    _, row = fixture(tmp_path, b'fictitious invalid PNG bytes', role)
    with pytest.raises(UnidentifiedImageError, match='cannot identify image file'):
        read(tmp_path, row, role)


def test_development_chunks_preserve_public_row_order_without_labels(tmp_path):
    rows = []
    for i, color in enumerate([(9, 1, 2), (8, 3, 4), (7, 5, 6)]):
        raw = png(color=color)
        path = tmp_path/f'{i}.png'; path.write_bytes(raw)
        rows.append(dict(path=path.name,sha256=hashlib.sha256(raw).hexdigest(),split='val'))
    chunks = development_images(tmp_path, rows, pixels, 2)
    assert [len(x) for x in chunks] == [2, 1]
    assert [x.tolist()[0][0] for x in torch.cat(chunks)] == [[9,1,2],[8,3,4],[7,5,6]]


def test_development_row_cannot_enter_training_reader(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('development role reached a training image read')
    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(ValueError, match='only train rows'):
        Images(tmp_path, [dict(split='val',path='unopened.png',sha256='0'*64)])
